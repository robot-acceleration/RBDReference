import numpy as np
import copy
np.set_printoptions(precision=4, suppress=True, linewidth = 100)

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj # instance of Robot Object class created by URDFparser``

    def cross_operator(self, v):
        # for any vector v, computes the operator v x 
        # vec x = [wx   0]
        #         [vox wx]
        v_cross = np.array([0, -v[2], v[1], 0, 0, 0,
                            v[2], 0, -v[0], 0, 0, 0,
                            -v[1], v[0], 0, 0, 0, 0,
                            0, -v[5], v[4], 0, -v[2], v[1], 
                            v[5], 0, -v[3], v[2], 0, -v[0],
                            -v[4], v[3], 0, -v[1], v[0], 0]
                          ).reshape(6,6)
        return(v_cross)
    
    def dual_cross_operator(self, v):
        return(-1 * self.cross_operator(v).T)

    def mxS(self, S, vec, alpha=1.0):
        # returns the spatial cross product between vectors S and vec. vec=[v0, v1 ... vn] and S = [s0, s1, s2, s3, s4, s5]
        # derivative of spatial motion vector = v x m
        """vecX = np.zeros((6))
        try:
            vecX[0] = vec[1]*alpha
            vecX[1] = -vec[0]*alpha
            vecX[3] = vec[4]*alpha
            vecX[4] = -vec[3]*alpha
        except:
            vecX[0] = vec[0,1]*alpha
            vecX[1] = -vec[0,0]*alpha
            vecX[3] = vec[0,4]*alpha
            vecX[4] = -vec[0,3]*alpha
        return vecX """
        return(alpha * np.dot(self.cross_operator(vec), S))        
    
    def fxv_simple(self, m, f):
        # force spatial vector cross product. 
        return(np.dot(self.dual_cross_operator(m), f))

    def fxv(self, fxVec, timesVec):
        # Fx(fxVec)*timesVec
        #   0  -v(2)  v(1)    0  -v(5)  v(4)
        # v(2)    0  -v(0)  v(5)    0  -v(3)
        #-v(1)  v(0)    0  -v(4)  v(3)    0
        #   0     0     0     0  -v(2)  v(1)
        #   0     0     0   v(2)    0  -v(0)
        #   0     0     0  -v(1)  v(0)    0
        result = np.zeros((6))
        result[0] = -fxVec[2] * timesVec[1] + fxVec[1] * timesVec[2] - fxVec[5] * timesVec[4] + fxVec[4] * timesVec[5]
        result[1] =  fxVec[2] * timesVec[0] - fxVec[0] * timesVec[2] + fxVec[5] * timesVec[3] - fxVec[3] * timesVec[5]
        result[2] = -fxVec[1] * timesVec[0] + fxVec[0] * timesVec[1] - fxVec[4] * timesVec[3] + fxVec[3] * timesVec[4]
        result[3] =                                                     -fxVec[2] * timesVec[4] + fxVec[1] * timesVec[5]
        result[4] =                                                      fxVec[2] * timesVec[3] - fxVec[0] * timesVec[5]
        result[5] =                                                     -fxVec[1] * timesVec[3] + fxVec[0] * timesVec[4]
        return result

    def fxS(self, S, vec, alpha = 1.0):
        # force spatial cross product with motion subspace 
        return -self.mxS(S, vec, alpha)

    def vxIv(self, vec, Imat):
        # necessary component in differentiating Iv (product rule).
        # We express I_dot x v as v x (Iv) (see Featherstone 2.14)
        # our core equation of motion is f = d/dt (Iv) = Ia + vx* Iv
        temp = np.matmul(Imat,vec)
        vecXIvec = np.zeros((6))
        vecXIvec[0] = -vec[2]*temp[1]   +  vec[1]*temp[2] + -vec[2+3]*temp[1+3] +  vec[1+3]*temp[2+3]
        vecXIvec[1] =  vec[2]*temp[0]   + -vec[0]*temp[2] +  vec[2+3]*temp[0+3] + -vec[0+3]*temp[2+3]
        vecXIvec[2] = -vec[1]*temp[0]   +  vec[0]*temp[1] + -vec[1+3]*temp[0+3] + vec[0+3]*temp[1+3]
        vecXIvec[3] = -vec[2]*temp[1+3] +  vec[1]*temp[2+3]
        vecXIvec[4] =  vec[2]*temp[0+3] + -vec[0]*temp[2+3]
        vecXIvec[5] = -vec[1]*temp[0+3] +  vec[0]*temp[1+3]
        return vecXIvec

    """
    Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

    RNEA divided into 3 parts: 
        1) calculate the velocity and acceleration of each body in the tree
        2) Calculate the forces necessary to produce these accelertions
        3) Calculate the forces transmitted across the joints from the forces acting on the bodies
    """

    
    def rnea_fpass(self, q, qd, qdd = None, GRAVITY = -9.81):
        """
        Forward Pass for RNEA algorithm. Computes the velocity and acceleration of each body in the tree necessary to produce a certain trajectory
        
        OUTPUT: 
        v : input qd is specifying value within configuration space with assumption of one degree of freedom. 
        Output velocity is in general body coordinates and specifies motion in full 6 degrees of freedom
        """
        assert len(q) == len(qd), "Invalid Trajectories"
        # not sure should equal num links or num joints. 
        assert len(q) == self.robot.get_num_joints(), "Invalid Trajectory, must specify coordinate for every body" 
        # allocate memory
        n = len(q)
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))

        gravity_vec = np.zeros((6)) # model gravity as a fictitious base acceleration. 
        # all forces subsequently offset by gravity. 
        gravity_vec[5] = -GRAVITY # a_base is gravity vec, linear in z direction

        # forward pass
        # vi = vparent + Si * qd_i
        # differentiate for ai = aparent + Si * qddi + Sdi * qdi
        
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) # the coordinate transform that brings into base reference frame
            S = self.robot.get_S_by_id(ind) # Subspace matrix
            # compute v and a
            if parent_ind == -1: # parent is base
                # v_base is zero so v[:,ind] remains 0
                a[:,ind] = np.matmul(Xmat,gravity_vec)
            else:
                v[:,ind] = np.matmul(Xmat,v[:,parent_ind]) # velocity of parent in base coordinates. 
                a[:,ind] = np.matmul(Xmat,a[:,parent_ind])
            v[:,ind] += S*qd[ind] # S turns config space into actual velocity
            a[:,ind] += self.mxS(S,v[:,ind],qd[ind])
            if qdd is not None:
                a[:,ind] += S*qdd[ind]

            # compute f
            Imat = self.robot.get_Imat_by_id(ind)
            f[:,ind] = np.matmul(Imat,a[:,ind]) + self.vxIv(v[:,ind],Imat)

        return (v,a,f)

    def rnea_bpass(self, q, qd, f, USE_VELOCITY_DAMPING = False):
        # allocate memory
        n = len(q) # assuming len(q) = len(qd)
        c = np.zeros(n)
        
        # backward pass
        # seek to calculate force transmitted from body i across joint i (fi) from the outside in.
        # fi = fi^B (net force) - fi^x (external forces, assumed to be known) - sum{f^j (all forces from children)}. 
        # Start with outermost as set of children is empty and go backwards to base.
        for ind in range(n-1,-1,-1):
            S = self.robot.get_S_by_id(ind)
            # compute c
            c[ind] = np.matmul(np.transpose(S),f[:,ind])
            # update f if applicable
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                temp = np.matmul(np.transpose(Xmat),f[:,ind])
                f[:,parent_ind] = f[:,parent_ind] + temp.flatten()

        # add velocity damping (defaults to 0)
        if USE_VELOCITY_DAMPING:
            for k in range(n):
                c[k] += self.robot.get_damping_by_id(k) * qd[k]

        return (c,f)

    def rnea(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
        """
        Recursive Newton-Euler Method is a recursive inverse dynamics algorithm to calculate the forces required for a specified trajectory

        RNEA divided into 3 parts: 
            1) calculate the velocity and acceleration of each body in the tree
            2) Calculate the forces necessary to produce these accelertions
            3) Calculate the forces transmitted across the joints from the forces acting on the bodies
            
        INPUT:
        q, qd, qdd: position, velocity, acceleration. Nx1 arrays where N is the number of bodies
        GRAVITY - gravitational field of the body; default is earth surface gravity, 9.81
        USE_VELOCITY_DAMPING: flag for whether velocity is damped, representing ___
        
        OUTPUTS: 
        c: Coriolis terms and other forces potentially be applied to the system. 
        v: velocity of each joint in world base coordinates rather than motion subspace
        a: acceleration of each joint in world base coordinates rather than motion subspace
        f: forces that joints must apply to produce trajectory
        """
        # forward pass
        (v,a,f) = self.rnea_fpass(q, qd, qdd, GRAVITY)
        # backward pass
        (c,f) = self.rnea_bpass(q, qd, f, USE_VELOCITY_DAMPING)

        return (c,v,a,f)

    def rnea_grad_fpass_dq(self, q, qd, v, a, GRAVITY = -9.81):
        
        # allocate memory
        n = len(qd)
        dv_dq = np.zeros((6,n,n))
        da_dq = np.zeros((6,n,n))
        df_dq = np.zeros((6,n,n))

        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){mxS(Xvp)}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dq[:,:,ind] = np.matmul(Xmat,dv_dq[:,:,parent_ind])
                dv_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,v[:,parent_ind]))
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(Xap)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dq[:,:,ind] = np.matmul(Xmat,da_dq[:,:,parent_ind])
            for c in range(n):
                da_dq[:,c,ind] += self.mxS(S,dv_dq[:,c,ind],qd[ind])
            if parent_ind != -1: # note that a_base is just gravity
                da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,a[:,parent_ind]))
            else:
                da_dq[:,ind,ind] += self.mxS(S,np.matmul(Xmat,gravity_vec))
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dq[:,:,ind] = np.matmul(Imat,da_dq[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dq[:,c,ind] += self.fxv(dv_dq[:,c,ind],Iv)
                df_dq[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dq[:,c,ind]))

        return (dv_dq, da_dq, df_dq)

    def rnea_grad_fpass_dqd(self, q, qd, v):
        
        # allocate memory
        n = len(qd)
        dv_dqd = np.zeros((6,n,n))
        da_dqd = np.zeros((6,n,n))
        df_dqd = np.zeros((6,n,n))

        # forward pass
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)
            # dv_du = X * dv_du_parent + (if c == ind){S}
            if parent_ind != -1: # note that v_base is zero so dv_du parent contribution is 0
                dv_dqd[:,:,ind] = np.matmul(Xmat,dv_dqd[:,:,parent_ind])
            dv_dqd[:,ind,ind] += S
            # da_du = x*da_du_parent + mxS_onCols(dv_du)*qd + (if c == ind){mxS(v)}
            if parent_ind != -1: # note that a_base is constant gravity so da_du parent contribution is 0
                da_dqd[:,:,ind] = np.matmul(Xmat,da_dqd[:,:,parent_ind])
            for c in range(n):
                da_dqd[:,c,ind] += self.mxS(S,dv_dqd[:,c,ind],qd[ind])
            da_dqd[:,ind,ind] += self.mxS(S,v[:,ind])
            # df_du = I*da_du + fx_onCols(dv_du)*Iv + fx(v)*I*dv_du
            Imat = self.robot.get_Imat_by_id(ind)
            df_dqd[:,:,ind] = np.matmul(Imat,da_dqd[:,:,ind])
            Iv = np.matmul(Imat,v[:,ind])
            for c in range(n):
                df_dqd[:,c,ind] += self.fxv(dv_dqd[:,c,ind],Iv)
                df_dqd[:,c,ind] += self.fxv(v[:,ind],np.matmul(Imat,dv_dqd[:,c,ind]))

        return (dv_dqd, da_dqd, df_dqd)

    def rnea_grad_bpass_dq(self, q, f, df_dq):
        
        # allocate memory
        n = len(q) # assuming len(q) = len(qd)
        dc_dq = np.zeros((n,n))
        
        for ind in range(n-1,-1,-1):
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            dc_dq[ind,:]  = np.matmul(np.transpose(S),df_dq[:,:,ind]) 
            # print("in bpass = dc_dq\n", dc_dq)
            # df_du_parent += X^T*df_du + (if ind == c){X^T*fxS(f)}
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                df_dq[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dq[:,:,ind])
                delta_dq = np.matmul(np.transpose(Xmat),self.fxS(S,f[:,ind]))
                for entry in range(6):
                    df_dq[entry,ind,parent_ind] += delta_dq[entry]

        return dc_dq

    def rnea_grad_bpass_dqd(self, q, df_dqd, USE_VELOCITY_DAMPING = False):
        
        # allocate memory
        n = len(q) # assuming len(q) = len(qd)
        dc_dqd = np.zeros((n,n))
        
        for ind in range(n-1,-1,-1):
            # dc_du is S^T*df_du
            S = self.robot.get_S_by_id(ind)
            dc_dqd[ind,:] = np.matmul(np.transpose(S),df_dqd[:,:,ind])
            # df_du_parent += X^T*df_du
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                df_dqd[:,:,parent_ind] += np.matmul(np.transpose(Xmat),df_dqd[:,:,ind]) 

        # add in the damping
        if USE_VELOCITY_DAMPING:
            for ind in range(n):
                dc_dqd[ind,ind] += self.robot.get_damping_by_id(ind)

        return dc_dqd

    def rnea_grad(self, q, qd, qdd = None, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False):
        # instead of passing in trajectory, what if we want our planning algorithm to solve for the optimal trajectory?
        """
        The gradients of inverse dynamics can be very extremely useful inputs into trajectory optimization algorithmss.
        Input: trajectory, including position, velocity, and acceleration
        Output: Computes the gradient of joint forces with respect to the positions and velocities. 
        """ 
        
        (c, v, a, f) = self.rnea(q, qd, qdd, GRAVITY)

        # forward pass, dq
        (dv_dq, da_dq, df_dq) = self.rnea_grad_fpass_dq(q, qd, v, a, GRAVITY)

        # forward pass, dqd
        (dv_dqd, da_dqd, df_dqd) = self.rnea_grad_fpass_dqd(q, qd, v)

        # backward pass, dq
        dc_dq = self.rnea_grad_bpass_dq(q, f, df_dq)
        # print("rnea_grad backward pass dc_dq\n", dc_dq)

        # backward pass, dqd
        dc_dqd = self.rnea_grad_bpass_dqd(q, df_dqd, USE_VELOCITY_DAMPING)

        dc_du = np.hstack((dc_dq,dc_dqd))
        return dc_du

    def minv_bpass(self, q):
        # allocate memory
        n = len(q)
        Minv = np.zeros((n,n))
        F = np.zeros((n,6,n))
        U = np.zeros((n,6))
        Dinv = np.zeros(n)

        # set initial IA to I
        IA = copy.deepcopy(self.robot.get_Imats_dict_by_id())
        
        # backward pass
        for ind in range(n-1,-1,-1):
            # Compute U, D
            S = self.robot.get_S_by_id(ind)
            subtreeInds = self.robot.get_subtree_by_id(ind)
            U[ind,:] = np.matmul(IA[ind],S)
            Dinv[ind] = 1/np.matmul(S.transpose(),U[ind,:])
            # Update Minv
            Minv[ind,ind] = Dinv[ind]
            for subInd in subtreeInds:
                Minv[ind,subInd] -= Dinv[ind] * np.matmul(S.transpose(),F[ind,:,subInd])
            # update parent if applicable
            parent_ind = self.robot.get_parent_id(ind)
            if parent_ind != -1:
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                # update F
                for subInd in subtreeInds:
                    F[ind,:,subInd] += U[ind,:]*Minv[ind,subInd]
                    F[parent_ind,:,subInd] += np.matmul(np.transpose(Xmat),F[ind,:,subInd]) 
                # update IA
                Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
                IaParent = np.matmul(np.transpose(Xmat),np.matmul(Ia,Xmat))
                IA[parent_ind] += IaParent

        return (Minv, F, U, Dinv)

    def minv_fpass(self, q, Minv, F, U, Dinv):
        n = len(q)
        
        # forward pass
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            S = self.robot.get_S_by_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            if parent_ind != -1:
                Minv[ind,ind:] -= Dinv[ind]*np.matmul(np.matmul(U[ind,:].transpose(),Xmat),F[parent_ind,:,ind:])
            
            F[ind,:,ind:] = np.outer(S,Minv[ind,ind:])
            if parent_ind != -1:
                F[ind,:,ind:] += np.matmul(Xmat,F[parent_ind,:,ind:])

        return Minv

    def minv(self, q, output_dense = True):
        # based on https://www.researchgate.net/publication/343098270_Analytical_Inverse_of_the_Joint_Space_Inertia_Matrix
        """ Computes the analytical inverse of the joint space inertia matrix
        CRBA calculates the joint space inertia matrix M to represent the composite inertia
        This is used in the fundamental motion equation M qdd + C = Tau
        Forward dynamics roughly calculates acceleration as M_inv ( Tau - C); analytic inverse - benchmark against Pinocchio
        """
        # backward pass
        (Minv, F, U, Dinv) = self.minv_bpass(q)

        # forward pass
        Minv = self.minv_fpass(q, Minv, F, U, Dinv)

        # fill in full matrix (currently only upper triangular)
        if output_dense:
            n = len(q)
            for col in range(n):
                for row in range(n):
                    if col < row:
                        Minv[row,col] = Minv[col,row]

        return Minv

    def crm(v):
        if len(v) == 6:
            vcross = np.array([0, -v[3], v[2], 0,0,0], [v[3], 0, -v[1], 0,0,0], [-v[2], v[1], 0, 0,0,0], [0, -v[6], v[5], 0,-v[3],v[2]], [v[6], 0, -v[4], v[3],0,-v[1]], [-v[5], v[4], 0, -v[2],v[1],0])
        else:
            vcross = np.array([0, 0, 0], [v[3], 0, -v[1]], [-v[2], v[1], 0])
        return vcross

    def aba_parallel(self, q, qd, tau, GRAVITY = -9.81):
       # allocate memory

        n = len(qd)
        v = np.zeros((6,n))
        c = np.zeros((6,n))
        a = np.zeros((6,n))
        d = np.zeros(n)
        U = np.zeros((6,n))
        u = np.zeros(n)
        IA = np.zeros((6,6,n))
        pA = np.zeros((6,n))
        qdd = np.zeros(n)
        S = np.zeros(n)
        

        n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0 
        
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        fext = np.ones(n)
 
        for i in range (0, n_bfs_levels): 
            inds = self.robot.get_ids_by_bfs_level(i)
            for ind in inds: 
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)
                if parent_ind == -1: # parent is base
                    v[:,ind] = S*qd[ind]
                
                else:
                    v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
                    v[:,ind] += S*qd[ind]
                    
                
        for ind in range (0, n): # in parallel
            Imat = self.robot.get_Imat_by_id(ind)
            IA[:,:,ind] = Imat

            vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
           [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0],
           [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
           [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]],
           [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
           [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

            crf=-np.transpose(vcross) #cross product of v x I and v x c can happen at same time in 2 diff parallel loops
            temp=np.matmul(crf,Imat)

            pA[:,ind]=np.matmul(temp,v[:,ind])[0]

            if self.robot.get_parent_id(ind):
                c[:,ind] = self.mxS(S,v[:,ind],qd[ind]) 
         
        for i in range (n_bfs_levels-1, -1,-1):
            inds = self.robot.get_ids_by_bfs_level(i)
            for ind in inds:
            #for ind in range (1, i):
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)
                U[:, ind] = IA[:,:,ind] @ S
                #print("U", U)
                d[ind] = S @ U[:, ind]
                #print("d",d)
                u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

                rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
                Ia = IA[:,:,ind] - rightSide

                temp = np.matmul(np.transpose(Xmat), Ia)
                if parent_ind != -1:
                    IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)  
        
        for ind in range(n): # in parallel 
            pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind] #compute pas in a diff parallel loop 
            temp1 = Xmat.T * fext[ind]
            pA[:, ind] += (temp1 @ pa)
        
        for i in range (0, n_bfs_levels):
            inds = self.robot.get_ids_by_bfs_level(i)
            for ind in inds: 
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                S = self.robot.get_S_by_id(ind)
                if parent_ind == -1: # parent is base
                    a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
                else:
                    a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

                temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
                qdd[ind] = temp / d[ind]
                a[:,ind] = a[:,ind] + qdd[ind]*S 

        return qdd 
    
    def aba(self, q, qd, tau, GRAVITY = -9.81):
        n = len(qd)
        v = np.zeros((6,n))
        c = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        d = np.zeros(n)
        U = np.zeros((6,n))
        u = np.zeros(n)
        IA = np.zeros((6,6,n))
        pA = np.zeros((6,n))
        qdd = np.zeros(n)
        
        
        gravity_vec = np.zeros((6))
        gravity_vec[5] = -GRAVITY # a_base is gravity vec 
                 
        for ind in range(n):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            S = self.robot.get_S_by_id(ind)

            if parent_ind == -1: # parent is base
                v[:,ind] = S*qd[ind]
                
            else:
                v[:,ind] = np.matmul(Xmat,v[:,parent_ind])
                v[:,ind] += S*qd[ind]
                c[:,ind] = self.mxS(S,v[:,ind],qd[ind])

            Imat = self.robot.get_Imat_by_id(ind)

            IA[:,:,ind] = Imat

            vcross=np.array([[0, -v[:,ind][2], v[:,ind][1], 0, 0, 0],
            [v[:,ind][2], 0, -v[:,ind][0], 0, 0, 0], 
            [-v[:,ind][1], v[:,ind][0], 0, 0, 0, 0],
            [0, -v[:,ind][5], v[:,ind][4], 0, -v[:,ind][2], v[:,ind][1]], 
            [v[:,ind][5],0, -v[:,ind][3], v[:,ind][2], 0, -v[:,ind][0]],
            [-v[:,ind][4], v[:,ind][3], 0, -v[:,ind][1], v[:,ind][0], 0]])

            crf=-np.transpose(vcross)
            temp=np.matmul(crf,Imat)

            pA[:,ind]=np.matmul(temp,v[:,ind])[0]
        
        for ind in range(n-1,-1,-1):
            S = self.robot.get_S_by_id(ind)
            parent_ind = self.robot.get_parent_id(ind)

            U[:,ind] = np.matmul(IA[:,:,ind],S)
            d[ind] = np.matmul(np.transpose(S),U[:,ind])
            u[ind] = tau[ind] - np.matmul(np.transpose(S),pA[:,ind])

            if parent_ind != -1:

                rightSide=np.reshape(U[:,ind],(6,1))@np.reshape(U[:,ind],(6,1)).T/d[ind]
                Ia = IA[:,:,ind] - rightSide

                pa = pA[:,ind] + np.matmul(Ia, c[:,ind]) + U[:,ind]*u[ind]/d[ind]

                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                temp = np.matmul(np.transpose(Xmat), Ia)

                IA[:,:,parent_ind] = IA[:,:,parent_ind] + np.matmul(temp,Xmat)

                temp = np.matmul(np.transpose(Xmat), pa)
                pA[:,parent_ind]=pA[:,parent_ind] + temp.flatten()
                                             
        for ind in range(n):

            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

            if parent_ind == -1: # parent is base
                a[:,ind] = np.matmul(Xmat,gravity_vec) + c[:,ind]
            else:
                a[:,ind] = np.matmul(Xmat, a[:,parent_ind]) + c[:,ind]

            S = self.robot.get_S_by_id(ind)
            temp = u[ind] - np.matmul(np.transpose(U[:,ind]),a[:,ind])
            qdd[ind] = temp / d[ind]
            a[:,ind] = a[:,ind] + qdd[ind]*S 
        
        return qdd 
    
    def crba( self, q, qd, tau):
                       
        #print("q", q)
        n = len(qd)
        
        C = self.rnea(q, qd, qdd = None, GRAVITY = -9.81)[0]
        
        IC = copy.deepcopy(self.robot.get_Imats_dict_by_id())# composite inertia calculation
        #print("IC", IC)
        M = np.zeros((n,n)) 
        alpha = np.zeros(len(IC))
        alpha = [(i,0) for i in alpha]

        beta = np.zeros(len(IC))
        beta = [(i,0) for i in alpha]

        j_ind = np.zeros(n)
        fh = np.zeros((n,6))

        n_bfs_levels = self.robot.get_max_bfs_level()
        #print("bfs levels", n_bfs_levels)
        for bfs_level in range(n_bfs_levels,0,-1):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            #print("bfs level", bfs_level)
            #print("inds len", len(inds))	
            #for parent_level in range(bfs_level):
                #par_inds = self.robot.get_ids_by_bfs_level(parent_level)
                
            for ind in inds: #this is parallel
                #print("ind:", ind)
                #print("parent ind:", parent_ind)
                #print("IC", IC)
                #print("IC[", ind, "] = ", IC[ind])
                #Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                #print("Xmat.T = ", Xmat.T)
                #print("Xmat = ", Xmat)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                alpha[ind] = Xmat.T@IC[ind]
                #print("alpha = ", alpha[ind])
            for ind in inds:
                #print("in beta loop")
                #print("IC", IC)
                #print("ind:", ind)
                parent_ind = self.robot.get_parent_id(ind)
                #Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                #Xmat = self.robot.get_Xmat_by_id(ind)
                #print("XMAT second loop", Xmat)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                #print("XMAT second loop", Xmat)
                #print("IC[parent_ind] = ", IC[parent_ind])
                beta[ind]= np.matmul(alpha[ind],Xmat)
                #print("beta[ind] = ", beta[ind])
                IC[parent_ind] += beta[ind]
                #print("parent_ind", parent_ind)
                #print("IC[parent_ind] = ", IC[parent_ind]) 

        # Calculation of fh and H[ind, ind] (second loop)
        """for ind in range(n-1, -1, -1): # in parallel
            #print("ind:", ind)
            S = self.robot.get_S_by_id(ind)
            #print("IC[ind] = ", IC[ind])
            fh = np.matmul(IC[ind], S)
            #print("fh", fh)
            H[ind, ind] = np.matmul(S, fh)
            #print("H[ind, ind]", H[ind, ind])
            j_ind[ind] = ind
          
        print("H", H)"""
        # Calculation of H[ind, j] and H[j, ind] (third loop)
        for ind in range(n-1, -1, -1): # in parallel
            #print("ind", ind)
            S = self.robot.get_S_by_id(ind)
            #print("S", S)
            fh[ind] = np.matmul(IC[ind],S)
            #print("IC", IC[ind])
            #print("fh", fh[ind])
        #print("fh ====== ")
        # print(fh)
        for ind in range(n-1, -1, -1): # in parallel
            S = self.robot.get_S_by_id(ind)
            M[ind, ind] = S.T@fh[ind]

        #print("H", H)
        # print(n_bfs_levels)
        for bfs_level in range(n_bfs_levels,0,-1):  # print out c++ in a python loop
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            #print("bfs_level)", bfs_level)
            #print("inds", inds)
            parent_inds = copy.deepcopy(inds) #parent_inds as temp array in c++
            # print(parent_inds)
            
            for parent_level in range(bfs_level): # print out c++ in a python loop
                # print("parent_level", parent_level)
                for parallel_index in range(len(inds)): # parallel loop
                    current_joint = inds[parallel_index]
                    # print("parallel_index", parallel_index)
                    # print("current joint", inds[parallel_index])
                    # print("current parent", parent_inds[parallel_index])
                    if parent_inds[parallel_index] != -1:
                #for ind in range(n-1, 0, -1):
                #for ind in inds:
                    #print("ind", ind)
                    #if j_ind[ind] != -1:
                        # get X
                        curr_parent = parent_inds[parallel_index]
                        # print("curr_parent real", curr_parent)
                        Xmat = self.robot.get_Xmat_Func_by_id(curr_parent)(q[curr_parent])

                        #print("fh pre", fh[current_joint])
                        #print("XMAT.T ", Xmat.T)
                        fh[current_joint] = Xmat.T@fh[current_joint]
                        # print("fh[curr_joint]", fh[current_joint])
                        # print("fh", fh)

                        # update parent
                        parent_inds[parallel_index] = self.robot.get_parent_id(curr_parent)
                        curr_parent = parent_inds[parallel_index]
                        # print("new parent", curr_parent)

                        #sync and new loop 

                #for ind in range(n-1, 0, -1):
                #for ind in inds:
                    #print("ind", ind)
                    #if j_ind[ind] != -1:
                        # get the parent S
                        S_parent = self.robot.get_S_by_id(curr_parent)
                        # compute H[i,j]
                        M[current_joint, curr_parent] = np.matmul(S_parent.T, fh[current_joint])
                        # print(M)
                        #print("H= S_parent.T, fh[current_joint]",H[current_joint, curr_parent])
                        M[curr_parent, current_joint] = M[current_joint, curr_parent] 
                        #print("H", H)
                    else:
                        print("HOW IN THE WORLD DID WE EVER GET HERE?")
        
        # print("fh", fh) 
        # print("fh.size", fh.size) robot


        """while self.robot.get_parent_id(j) > -1:
                Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                fh = np.matmul(Xmat.T, fh)
                j = self.robot.get_parent_id(j)
                S = self.robot.get_S_by_id(j)
                H[ind, j] = np.matmul(S.T, fh)
                H[j, ind] = H[ind, j]"""

        """for ind in range(n-1,-1,-1):
            parent_ind = self.robot.get_parent_id(ind)
            Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])

            if parent_ind != -1:
                IC[parent_ind] = IC[parent_ind] + np.matmul(np.matmul(Xmat.T, IC[ind]),Xmat)

        H = np.zeros((n,n))

        for ind in range(n):

            S = self.robot.get_S_by_id(ind)
            fh = np.matmul(IC[ind],S)
            H[ind,ind] = np.matmul(S,fh)
            j = ind

            while self.robot.get_parent_id(j) > -1:
                Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                fh = np.matmul(Xmat.T,fh)
                j = self.robot.get_parent_id(j)
                S = self.robot.get_S_by_id(j)
                H[ind,j] = np.matmul(S.T, fh)
                H[j,ind] = H[ind,j]

        sub=np.subtract(tau,C) """
        return M
    
    def crba_parallel( self, q, qd, tau):
        n = len(qd)
        C = self.rnea(q, qd, qdd=None, GRAVITY=-9.81)[0]

        IC = copy.deepcopy(self.robot.get_Imats_dict_by_id()) 
        M = np.zeros((n, n))

        alpha = np.zeros(len(IC))
        alpha = [(i,0) for i in alpha]

        n_bfs_levels = self.robot.get_max_bfs_level()
        for bfs_level in range(n_bfs_levels,0,-1):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)	
            for ind in inds: #this is parallel
                #print("ind:", ind)
                #print("parent ind:", parent_ind)
                #print("IC[", ind, "] = ", IC[ind])
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                #print("Xmat.T = ", Xmat.T)
                #print("Xmat = ", Xmat)
                alpha[ind] = Xmat.T@IC[ind]
                #print("alpha = ", alpha[ind])
            for ind in inds:
                #print("ind:", ind)
                parent_ind = self.robot.get_parent_id(ind)
                Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
                #print("IC[parent_ind] = ", IC[parent_ind])
                IC[parent_ind] += np.matmul(alpha[ind],Xmat)
                #print("IC[parent_ind] = ", IC[parent_ind])

        # Calculation of fh and H[ind, ind]
        for ind in range(n-1, -1, -1): # in parallel
            S = self.robot.get_S_by_id(ind)
            fh = np.matmul(IC[ind], S)
            M[ind, ind] = np.matmul(S, fh)
                    
        # Calculation of H[ind, j] and H[j, ind]
        for ind in range(n-1, -1, -1): # in parallel
            S = self.robot.get_S_by_id(ind)
            fh = np.matmul(IC[ind],S)
            j = ind
            while self.robot.get_parent_id(j) > -1:
                Xmat = self.robot.get_Xmat_Func_by_id(j)(q[j])
                fh = np.matmul(Xmat.T, fh)
                j = self.robot.get_parent_id(j)
                S = self.robot.get_S_by_id(j)
                M[ind, j] = np.matmul(S.T, fh)
                M[j, ind] = M[ind, j]
        print("fh \n", fh)
                            
        return M
    
    def icrf(self, v):
        #helper function defined in spatial_v2_extended library, called by idsva()
        v = np.array(v).flatten()
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asmatrix(res)
    
    def idsva(self, q, qd, qdd, GRAVITY = -9.81):
        """alternative to rnea_grad(), described in "Efficient Analytical Derivatives of Rigid-Body Dynamics using
Spatial Vector Algebra" (Singh, Russel, and Wensing)

        :param q: initial joint positions
        :type q: array representing a 1 by n vector, where n is the number of joints
        :param qd: initial joint velocities
        :type qd: array representing a 1 by n vector
        :param qdd: desired joint accelerations 
        :type qdd: array representing a 1 by n vector
        :param GRAVITY: defaults to -9.81
        :type GRAVITY: float, optional
        :return: dtau_dq, dtau_dqd-- the gradient of the resulting torque with respect to initial joint position and velocity
        :rtype: list
        """
        # allocate memory
        n = len(qd)
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame 
        Xdown0 = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        Sdd = np.zeros((6,n))
        Sj = np.zeros((6,n)) 
        IC = [None] * n 
        BC  = [None] * n 
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        
        # forward pass
        for i in range(n):
            parent_i = self.robot.get_parent_id(i)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])

            # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i])

            S[:,i] = self.robot.get_S_by_id(i)

            S[:,i] = Xdown0[i] @ S[:,i]
            Sd[:,i] = self.cross_operator(v[:,i]) @ S[:,i]
            Sdd[:,i]= self.cross_operator(a[:,i])@ S[:,i]
            Sdd[:,i] = Sdd[:,i] + self.cross_operator(v[:,i])@ Sd[:,i]
            Sj[:,i] = 2*Sd[:,i] + self.cross_operator(S[:,i]*qd[i])@ S[:,i]

            v6x6 = self.cross_operator(v[:,i])
            v[:,i] = v[:,i] + S[:,i]*qd[i]
            a[:,i] = a[:,i] + np.array(v6x6 @ S[:,i]*qd[i])

            if qdd is not None:
                a[:,i] += S[:,i]*qdd[i]
            
            # compute f, IC, BC
            Imat = self.robot.get_Imat_by_id(i)

            IC[i] = np.array(Xup0[i]).T  @ (Imat @ Xup0[i])
            f[:,i] = IC[i] @ a[:,i] + self.dual_cross_operator(v[:,i]) @ IC[i] @ v[:,i]
            f[:,i] = np.asarray(f[:,i]).flatten()
            BC[i] = (self.dual_cross_operator(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.cross_operator(v[:,i]))
        

        t1 = np.zeros((6,n))
        t2 = np.zeros((6,n))
        t3 = np.zeros((6,n))
        t4 = np.zeros((6,n))
        dtau_dq = np.zeros((n,n))
        dtau_dqd = np.zeros((n,n))


        #backward pass
        for i in range(n-1,-1,-1):

            t1[:,i] = IC[i] @ S[:,i]
            t2[:,i] = BC[i] @ S[:,i].T + IC[i] @ Sj[:,i]
            t3[:,i] = BC[i] @ Sd[:,i] + IC[i] @ Sdd[:,i] + self.icrf(f[:,i]) @ S[:,i]
            t4[:,i] = BC[i].T @ S[:,i]

            subtree_ids = self.robot.get_subtree_by_id(i) #list of all subtree ids (inclusive)



            dtau_dq[i, subtree_ids[1:]] = S[:,i] @ t3[:,subtree_ids[1:]]
            
            dtau_dq[subtree_ids[0:], i] = Sdd[:,i] @ t1[:,subtree_ids[0:]] + \
                                        Sd[:,i] @ t4[:,subtree_ids[0:]] 
            
            dtau_dqd[i, subtree_ids[1:]] = S[:,i] @ t2[:,subtree_ids[1:]]

            dtau_dqd[subtree_ids[0:], i] = Sj[:,i] @ t1[:,subtree_ids[0:]] + \
                                        S[:,i] @ t4[:,subtree_ids[0:]] 

            p = self.robot.get_parent_id(i)
            if p >= 0:
                IC[p] = IC[p] + IC[i]
                BC[p] = BC[p] + BC[i]
                f[:,p] = f[:,p] + f[:,i]


        return dtau_dq, dtau_dqd
    
    def __init__(self, robotObj):
        self.robot = robotObj
        
    def crm(self, v): 
        vcross = [[   0, -v[2],  v[1],    0.,    0.,    0.],
                  [v[2],    0., -v[0],    0.,    0.,    0.],
                  [-v[1], v[0],    0.,    0.,    0.,    0.],
                  [  0., -v[5],  v[4],    0., -v[2],  v[1]],
                  [v[5],    0., -v[3],  v[2],    0., -v[0]],
                  [-v[4], v[3],    0., -v[1],  v[0],    0.]]

        return np.asmatrix(vcross)

    def crf(self, v):
        vcross = -self.crm(v).conj().T #negative complex conjugate transpose
        return vcross

    def icrf(self, v):
        v = np.array(v).flatten()
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asmatrix(res)
    
    def dot_matrix(self, I, v):
        A =  self.crf(v) @ I - I @ self.crm(v)
        scale_factor = 10**-15
        A = A / scale_factor
        return self.crf(v) @ I - I @ self.crm(v)

    def second_order_idsva_series(self, q, qd, qdd, GRAVITY = -9.81):
        # allocate memory
        n = len(qd) # n = 7
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame
        Xdown0 = [None] * n
        IC = [None] * n
        BC = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        vJ = np.zeros((6,n))
        aJ = np.zeros((6,n))
        psid = np.zeros((6,n))
        psidd = np.zeros((6,n))
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec

        # forward pass 
        modelNB = n
        modelNV = self.robot.get_num_joints()
        for i in range(modelNB):
            parent_i = self.robot.get_parent_id(i)
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])
          # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec # 
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i]) 
            S[:,i] = self.robot.get_S_by_id(i)
            S[:,i] = Xdown0[i] @ S[:,i]
            vJ[:,i] = S[:,i] * qd[i]
            aJ[:,i] = self.crm(v[:,i])@vJ[:,i] + S[:,i] * qdd[i]
            psid[:,i] = self.crm(v[:,i])@S[:,i]
            psidd[:,i] = self.crm(a[:,i])@S[:,i] + self.crm(v[:,i])@psid[:,i]
            v[:,i] = v[:,i] + vJ[:,i]
            a[:,i] = a[:,i] + aJ[:,i]
            I = self.robot.get_Imat_by_id(i)
            IC[i] = np.array(Xup0[i]).T @ (I @ Xup0[i])
            Sd[:, i] = self.crm(v[:,i]) @ S[:,i]
            assert Sd[:, i].shape == (6,), f"Unexpected shape for Sd[:, {i}]: {Sd[:, i].shape}"
            BC[i] = (self.crf(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.crm(v[:,i]))
            f[:,i] = IC[i] @ a[:,i] + self.crf(v[:,i]) @ IC[i] @v[:,i] 
        
        # Matrix Intialization
        dM_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dq = np.zeros((modelNV,modelNV,modelNV))
        d2tau_dqd = np.zeros((modelNV,modelNV,modelNV))
        d2tau_cross = np.zeros((modelNV,modelNV,modelNV))

        #backward pass
        for i in range(modelNB-1,-1,-1):
            modelnv = 1 # DOF of ith joint, Since revolute type for iiwa so this is 1
            for p in range(modelnv):
                S_p = S[:, i]  
                Sd_p = Sd[:, i]
                psid_p = psid[:, i]
                psidd_p = psidd[:, i]
                
                Bic_phii = self.icrf(IC[i] @ S_p)
                Bic_psii_dot = 2 * 0.5 * (self.crf(psid_p) @ IC[i] + self.icrf(IC[i] @ psid_p) - IC[i] @ self.crm(psid_p))

                A0 = self.icrf(IC[i] @ S_p )
                A1 = self.dot_matrix(IC[i], S_p) 
                A2 = 2 * A0 - Bic_phii
                A3 = Bic_psii_dot + self.dot_matrix(BC[i], S_p)
                A4 = self.icrf(BC[i].T @ S_p)
                A5 = self.icrf(BC[i] @ psid_p  + IC[i]@psidd_p + self.crf(S_p) @ f[:, i])
                A6 = self.crf(S_p) @ IC[i] + A0
                A7 = self.icrf(BC[i] @ S_p + IC[i] @ (psid_p + Sd_p) )
                ii = i
                j = i 

                while j >= 0 :
                    jj = j 
                    modelnvj = 1 # DOF of jth joint
                    for t in range(modelnvj):
                        S_t = S[:, j]
                        Sd_t = Sd[:, j]
                        psid_t = psid[:, j]
                        psidd_t = psidd[:, j]    
                        u1 = A3.T @ S_t
                        u2 = A1.T @ S_t
                        u3 = A3 @ psid_t + A1 @ psidd_t + A5 @ S_t
                        u4 = A6 @ S_t
                        u5 = A2 @ psid_t + A4 @ S_t
                        u6 = Bic_phii @ psid_t + A7 @ S_t
                        u7 = A3 @ S_t + A1 @ (psid_t + Sd_t)
                        u8 = A4 @ S_t - Bic_phii.T @ psid_t
                        u9 = A0 @ S_t
                        u10 = Bic_phii @ S_t
                        u11 = Bic_phii.T @ S_t
                        u12 = A1 @ S_t
                        
                        k = j
                        while k >= 0:
                            kk = k 
                            modelnvk = 1 # DOF of kth joint
                            for r in range(modelnvk):
                                S_r = S[:, k]
                                Sd_r = Sd[:, k]
                                psid_r = psid[:, k]
                                psidd_r = psidd[:, k]
                                p1 = psid_r @ u11.T
                                p2 =  psid_r @ u8.T + psidd_r @ u9.T
                                d2tau_dq[ii, jj, kk] = p2
                                d2tau_cross[ii, kk, jj] = -p1
                                # print(f"The value of d2tau_cross[{ii+1}, {kk+1}, {jj+1}]: ")
                                # print(f"Printing -p1 = {-p1} \n" )

                                if j != i:
                                    d2tau_dq[jj, kk, ii] = psid_r @ u1.T + psidd_r @ u2.T
                                    # print(f"The value of d2tau_dq[{jj+1}, {kk+1}, {ii+1}]: ")
                                    # print(f"Printing psid_r @ u1.T + psidd_r @ u2.T = {psid_r @ u1.T + psidd_r @ u2.T} \n" )
                                    d2tau_dq[jj, ii, kk] = d2tau_dq[jj, kk, ii]
                                    # print(f"The value of d2tau_dq[{jj+1}, {ii+1}, {kk+1}]: ")
                                    # print(f"Printing d2tau_dq[jj, kk, ii] \n" )
                                    d2tau_cross[jj, kk, ii] = p1
                                    # print(f"The value of d2tau_cross[{jj+1}, {kk+1}, {ii+1}]: ")
                                    # print(f"Printing p1 = {p1} \n" )
                                    d2tau_cross[jj, ii, kk] = S_r @ u1.T + (psid_r + Sd_r) @ u2.T
                                    # print(f"The value of d2tau_cross[{jj+1}, {ii+1}, {kk+1}]: ")
                                    # print(f"Printing S_r @ u1.T + (psid_r + Sd_r) @ u2.T = {S_r @ u1.T + (psid_r + Sd_r) @ u2.T} \n" )
                                    d2tau_dqd[jj, kk, ii] = S_r @ u11.T
                                    # print(f"The value of d2tau_dqd[{jj+1}, {kk+1}, {ii+1}]: ")
                                    # print(f"Printing S_r @ u11.T = {S_r @ u11.T} \n" )
                                    d2tau_dqd[jj, ii, kk] = d2tau_dqd[jj, kk, ii]
                                    # print(f"The value of d2tau_dqd[{jj+1}, {ii+1}, {kk+1}]: ")
                                    # print(f"Printing d2tau_dqd[jj, kk, ii] \n" )
                                    dM_dq[kk, jj, ii] = u12 @ S_r.T
                                    # print(f"The value of dM_dq[{kk+1}, {jj+1}, {ii+1}]: ")
                                    # print(f"Printing u12 @ S_r.T = {u12 @ S_r.T} \n" )
                                    dM_dq[jj, kk, ii] = u12 @ S_r.T
                                    # print(f"The value of dM_dq[{jj+1}, {kk+1}, {ii+1}]: ")
                                    # print(f"Printing u12 @ S_r.T = {u12 @ S_r.T} \n" )

                                if k != j:
                                    d2tau_dq[ii, kk, jj] = p2
                                    # print(f"The value of d2tau_dq[{ii+1}, {kk+1}, {jj+1}]: ")
                                    # print(f"Printing p2 = {p2} \n" )
                                    d2tau_dq[kk, ii, jj] = u3 @ S_r.T
                                    # print(f"The value of d2tau_dq[{kk+1}, {ii+1}, {jj+1}]: ")
                                    # print(f"Printing u3 @ S_r.T = {u3 @ S_r.T} \n" )
                                    d2tau_dqd[ii, jj, kk] = -S_r @ u11.T
                                    # print(f"The value of d2tau_dqd[{ii+1}, {jj+1}, {kk+1}]: ")
                                    # print(f"Printing -S_r @ u11.T = {-S_r @ u11.T} \n" )
                                    d2tau_dqd[ii, kk, jj] = -S_r @ u11.T
                                    # print(f"The value of d2tau_dqd[{ii+1}, {kk+1}, {jj+1}]: ")
                                    # print(f"Printing -S_r @ u11.T = {-S_r @ u11.T} \n" )
                                    d2tau_cross[ii, jj, kk] = u5 @ S_r.T + (psid_r + Sd_r) @ u9.T
                                    # print(f"The value of d2tau_cross[{ii+1}, {jj+1}, {kk+1}]: ")
                                    # print(f"Printing u5 @ S_r.T + (psid_r + Sd_r) @ u9.T = {u5 @ S_r.T + (psid_r + Sd_r) @ u9.T} \n" )
                                    d2tau_cross[kk, jj, ii] = u6 @ S_r.T
                                    # print(f"The value of d2tau_cross[{kk+1}, {jj+1}, {ii+1}]: ")
                                    # print(f"Printing u6 @ S_r.T = {u6 @ S_r.T} \n" )
                                    dM_dq[kk, ii, jj] = u9 @ S_r.T
                                    # print(f"The value of dM_dq[{kk+1}, {ii+1}, {jj+1}]: ")
                                    # print(f"Printing u9 @ S_r.T = {u9 @ S_r.T} \n" )
                                    dM_dq[ii, kk, jj] = u9 @ S_r.T
                                    # print(f"The value of dM_dq[{ii+1}, {kk+1}, {jj+1}]: ")
                                    # print(f"Printing u9 @ S_r.T = {u9 @ S_r.T} \n" )
                                
                                    if j != i:
                                        d2tau_dq[kk, jj, ii] = d2tau_dq[kk, ii, jj]
                                        # print(f"The value of d2tau_dq[{kk+1}, {jj+1}, {ii+1}]: ")
                                        # print(f"Printing d2tau_dq[kk, ii, jj] \n" )
                                        d2tau_dqd[kk, ii, jj] = u10 @ S_r.T
                                        # print(f"The value of d2tau_dqd[{kk+1}, {ii+1}, {jj+1}]: ")
                                        # print(f"Printing u10 @ S_r.T = {u10 @ S_r.T} \n" )
                                        d2tau_dqd[kk, jj, ii] = d2tau_dqd[kk, ii, jj]
                                        # print(f"The value of d2tau_dqd[{kk+1}, {jj+1}, {ii+1}]: ")
                                        # print(f"Printing d2tau_dqd[kk, ii, jj] \n" )
                                        d2tau_cross[kk, ii, jj] = u7 @ S_r.T
                                        # print(f"The value of d2tau_cross[{kk+1}, {ii+1}, {jj+1}]: ")
                                        # print(f"Printing u7 @ S_r.T = {u7 @ S_r.T} \n" )
                                    else:
                                        d2tau_dqd[kk,jj,ii] = u4 @ S_r.T
                                        # print(f"The value of d2tau_dqd[{kk+1},{jj+1},{ii+1}]: ")
                                        # print(f"Printing u4 @ S_r.T = {u4 @ S_r.T} \n" )
                                else:
                                    d2tau_dqd[ii,jj,kk] = - S_r @ u2.T
                                    # print(f"The value of d2tau_dqd[{ii+1},{jj+1},{kk+1}]: ")
                                    # print(f"Printing - S_r @ u2.T = {- S_r @ u2.T} \n" )
                            g = self.robot.get_parent_id(k)
                            # print(f"The value of k = {g+1}")
                            k = g
                    z = self.robot.get_parent_id(j)
                    # print(f"The value of j = {z+1}")
                    j = z
            pi = self.robot.get_parent_id(i)
            if pi >= 0:
                IC[pi] = IC[pi] + IC[i]
                BC[pi] = BC[pi] + BC[i]
                f[:, pi] = f[:, pi] + f[:, pi + 1]
        return d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq
            
    
    def fdsva( self, q, qd, qdd, tau, id_M, iddq, iddqd):

        # qdd[0] = fd(q,qd,tau)
        n = len(qd)
        fddq = np.zeros((n,n))
        fddqd = np.zeros((n,n))
        fddt = np.zeros((n,n))
        Minv = self.minv(q)

        fddq = (-1)*Minv@iddq
        fddqd = (-1)*Minv@iddqd
        fddt = Minv
       
        return fddq, fddqd, fddt 
    
    
   
    def finite_diff_fdsva(self, q, qd, qdd, tau):

        fd_q = np.zeros((len(q),len(q)))
        fd_qd = np.zeros((len(qd),len(qd)))
        h = 0.01 #needs to be changed based on dist between points on grid

        for i in range(len(q)):
            pos_plus = np.copy(q)
            pos_minus = np.copy(q)

            pos_plus[i] = pos_plus[i] + h
            pos_minus[i] = pos_minus[i] - h

            # fd_plus_c, fd_plus_v, fd_plus_a, fd_plus_f = self.rnea(pos_plus, qd)
            # fd_minus_c, fd_minus_v, fd_minus_a, fd_minus_f = self.rnea(pos_minus, qd)
            fd_plus_c = self.aba(pos_plus,qd,tau)
            fd_minus_c = self.aba(pos_minus,qd,tau)

            c = fd_plus_c - fd_minus_c
            fd_temp = (c) / (2*h) 
            for j,temp in enumerate(fd_temp):
                fd_q[i][j] = temp
        
        for i in range(len(qd)):

            vel_plus = np.copy(qd)
            vel_minus = np.copy(qd)

            vel_plus[i] = vel_plus[i] + h
            vel_minus[i] = vel_minus[i] - h

            # fd_plus_c, fd_plus_v, fd_plus_a, fd_plus_f = self.rnea(q, vel_plus)
            # fd_minus_c, fd_minus_v, fd_minus_a, fd_minus_f = self.rnea(q, vel_minus)
            fd_plus_c = self.aba(q,vel_plus,tau)
            fd_minus_c = self.aba(q,vel_minus,tau)

            fd_temp = (fd_plus_c - fd_minus_c) / (2 * h)
            for j,temp in enumerate(fd_temp):
                fd_qd[i][j] = temp

        return fd_q, fd_qd
    
    def finite_diff_idsvaso(self, q, qd, qdd, tau):
        h = 0.1
        n = len(q)

        d2tau_dq = np.zeros((n,n,n))
        d2tau_dqd = np.zeros((n,n,n))
        d2tau_cross = np.zeros((n,n,n))
        dM_dq = np.zeros((n,n,n))

        for i in range(n):
            q_plus = np.copy(q)
            q_minus = np.copy(q)
            q_plus[i] = q_plus[i] + h    
            q_minus[i] = q_minus[i] - h  

            iddq_p, iddqd_p = np.array(self.idsva(q_plus,qd, qdd))
            iddq_m, iddqd_m = np.array(self.idsva(q_minus,qd, qdd))

            # print("iddq_p", iddq_p)
            # print("iddqd_p", iddqd_p)

            # print("iddq_m", iddq_m)
            # print("iddqd_m", iddqd_m)


            d2tau_dq[:,:,i] = np.squeeze((iddq_p - iddq_m) / (2*h))
            
        for i in range(n):
            qd_plus = np.copy(qd)
            qd_minus = np.copy(qd)

            qd_plus[i] = qd_plus[i] + h    
            qd_minus[i] = qd_minus[i] - h  
            iddq_p, iddqd_p = self.idsva(q,qd_plus,qdd)
            iddq_m, iddqd_m = self.idsva(q,qd_minus,qdd)

            d2tau_dqd[:,:,i] = np.squeeze((iddqd_p - iddqd_m) / (2*h))

        for i in range(n):
            q_plus = np.copy(q)
            q_minus = np.copy(q)
            qd_plus = np.copy(qd)
            qd_minus = np.copy(qd)

            q_plus[i] = q_plus[i] + h    
            q_minus[i] = q_minus[i] - h 
            iddq_p, iddqd_p = self.idsva(q_plus,qd,qdd)
            iddq_m, iddqd_m = self.idsva(q_minus,qd,qdd)
    
            d2tau_cross[:,:,i] = np.squeeze((iddqd_p - iddqd_m) / (2*h))
            
        for i in range(n):
            q_plus = np.copy(q)
            q_minus = np.copy(q)

            q_plus[i] = q_plus[i] + h    
            q_minus[i] = q_minus[i] - h 
            m_p = self.crba(q_plus,qd,tau)
            m_m = self.crba(q_minus,qd,tau)
     
            dM_dq[:,:,i] = np.squeeze((m_p - m_m) / (2*h))

        return d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq


    def finite_diff_fdsvaso1(self, q, qd, tau):
        h = 0.001
        q_qd__tau_plus = np.hstack((q,qd,tau))
        q_qd__tau_minus = np.hstack((q,qd,tau))
        fd_output = []

        for ind in range(21):
        
            # print(ind)
            # print(q_qd__tau_plus)

            q_qd__tau_plus[ind] += h
            # print("q_qd__tau_plus", q_qd__tau_plus)
            minv_plus = self.minv(q_qd__tau_plus[0:len(q)])
            # print("minv_plus", minv_plus)
            qdd_plus = self.aba(q_qd__tau_plus[0:len(q)], q_qd__tau_plus[len(q):len(q)+len(qd)], q_qd__tau_plus[len(q)+len(qd):], GRAVITY = -9.81)
            # print("qdd_plus", qdd_plus)
            dc_du_plus = self.rnea_grad(q_qd__tau_plus[0:len(q)], q_qd__tau_plus[len(q):len(q)+len(qd)], qdd_plus, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False)
            dc_dq_plus = dc_du_plus[:,:7]
            # print("dc_dq_plus", dc_dq_plus)
            dc_dqd_plus = dc_du_plus[:,7:]
            # print("dc_dqd_plus", dc_dqd_plus)
            df_du_plus = self.fdsva(q_qd__tau_plus[0:len(q)], q_qd__tau_plus[len(q):len(q)+len(qd)], qdd_plus, q_qd__tau_plus[len(q)+len(qd):], minv_plus, dc_dq_plus, dc_dqd_plus)
            # print("df_du_plus", df_du_plus)


            q_qd__tau_minus[ind] -= h
            # print("q_qd__tau_minus", q_qd__tau_minus)
            minv_minus = self.minv(q_qd__tau_minus[0:len(q)])
            # print("minv_minus", minv_minus)
            qdd_minus = self.aba(q_qd__tau_minus[0:len(q)], q_qd__tau_minus[len(q):len(q)+len(qd)], q_qd__tau_minus[len(q)+len(qd):], GRAVITY = -9.81)
            # print("qdd_minus", qdd_minus)
            dc_du_minus = self.rnea_grad(q_qd__tau_minus[0:len(q)], q_qd__tau_minus[len(q):len(q)+len(qd)], qdd_minus, GRAVITY = -9.81, USE_VELOCITY_DAMPING = False)
            dc_dq_minus = dc_du_minus[:,:7]
            # print("dc_dq_minus", dc_dq_minus)
            dc_dqd_minus = dc_du_minus[:,7:]
            # print("dc_dqd_minus", dc_dqd_minus)
            df_du_minus = self.fdsva(q_qd__tau_minus[0:len(q)], q_qd__tau_minus[len(q):len(q)+len(qd)], qdd_minus, q_qd__tau_minus[len(q)+len(qd):], minv_minus, dc_dq_minus, dc_dqd_minus)
            # print("df_du_minus", df_du_minus)

            plus = np.concatenate((df_du_plus[0], df_du_plus[1], df_du_plus[2]), axis=1)
            # print("plus", plus)
            minus = np.concatenate((df_du_minus[0], df_du_minus[1], df_du_minus[2]), axis=1)
            # print("minus", minus)
            temp = np.zeros((7,21))

            for i in range(7):
                for j in range(21):
                    temp[i][j] = (plus[i][j] - minus[i][j] )/ (2*h)
            # print("temp", temp)
            fd_output.append(temp)
            
            q_qd__tau_plus[ind] -= h
            q_qd__tau_minus[ind] += h

        return fd_output
             


    # def finite_diff_fdsvaso(self, q, qd, qdd, tau, id_M, dtau_dq, dtau_dqd):
    #     h = 0.01

    #     fdsva = self.fdsva(q, qd, qdd, tau, id_M, dtau_dq, dtau_dqd)
    #     input_fddq, input_fddqd, input_fddt = fdsva
        
    #     n = len(input_fddq)
    #     m = len(input_fddq[0])

    #     output_fddqq = [] #np.zeros((7,7,7))
    #     output_fddqd_dq = [] #np.zeros((7,7,7))
    #     output_fddqdqd = [] #np.zeros((7,7,7))
    #     output_fddtdq = [] #np.zeros((7,7,7))
  
        
    #     #plus and minus for dtau_dq and dtau_dqd --> bc need to change q and qd for idsva so intead of putting dtau_dq maybe call idsva(q_plus)... 
    #     for i in range(n):
    #         q_plus = np.copy(q)
    #         q_minus = np.copy(q)
    #         # print(q_plus)

    #         q_plus[i] = q_plus[i] + h    
    #         q_minus[i] = q_minus[i] - h  
    #         # print("q-plus", q_plus)
    #         iddq_p, iddqd_p = self.idsva(q,qd,qdd)
    #         iddq_m, iddqd_m = self.idsva(q,qd,qdd)
    #         fddq_plus, _, _ = self.fdsva(q_plus,qd,qdd,tau,id_M,iddq_p, iddqd_p) 
    #         # print("fddq_plus", fddq_plus)  
    #         fddq_minus, _, _ = self.fdsva(q_minus,qd,qdd,tau,id_M,iddq_m, iddqd_m)  
    #         # print("fddq_minus", fddq_minus)  

    #         # fddq_plus = [[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],[21,22,23,24,25,26,27],[28,29,30,31,32,33,34],[35,36,37,38,39,40,41],[42,43,44,45,46,47,48]]
    #         temp = np.copy(fddq_plus) #np.zeros((n,m))
    #         for j in range(n):
    #             temp[i][j] = (fddq_plus[i][j] - fddq_minus[i][j]) / (2*h)
            
    #         output_fddqq.append(temp)
    #     print("output_fddqq", output_fddqq)
        
    #     for i in range(n):
    #         q_plus = np.copy(q)
    #         q_minus = np.copy(q) 

    #         q_plus[i] = q_plus[i] + h    
    #         q_minus[i] = q_minus[i] - h 

    #         iddq_p, iddqd_p = self.idsva(q_plus,qd,qdd)
    #         iddq_m, iddqd_m = self.idsva(q_minus,qd,qdd)
    #         _, fddqd_plus, _ = self.fdsva(q_plus,qd,qdd,tau,id_M,iddq_p, iddqd_p)   
    #         _, fddqd_minus, _ = self.fdsva(q_minus,qd,qdd,tau,id_M,iddq_m, iddqd_m)  

    #         temp = np.zeros((n, n))
    #         for j in range(temp.shape[0]):
    #             for k in range(temp.shape[0]):
    #                 temp[j][k] = (fddqd_plus[j][k] - fddqd_minus[j][k]) / (2*h)
            
    #         output_fddqd_dq.append(temp)
    #     # print("output_fddqd_dq", output_fddqd_dq)


    #     for i in range(n):
    #         qd_plus = np.copy(qd)
    #         qd_minus = np.copy(qd)
    #         print("qd_plus", qd_plus)

    #         qd_plus[i] = q_plus[i] + h    
    #         qd_minus[i] = q_minus[i] - h   

    #         iddq_p, iddqd_p = self.idsva(q,qd_plus,qdd)
    #         iddq_m, iddqd_m = self.idsva(q,qd_minus,qdd)
    #         _, fddqd_plus, _ = self.fdsva(q,qd_plus,qdd,tau,id_M,iddq_p, iddqd_p)   
    #         _, fddqd_minus, _ = self.fdsva(q,qd_minus,qdd,tau,id_M,iddq_m, iddqd_m)  

    #         temp = np.zeros((n,n))
    #         for j in range(n):
    #             temp[i][j] = (fddqd_plus[i][j] - fddqd_minus[i][j]) / (2*h)
            
    #         output_fddqdqd.append(temp)
    #     # print("output_fddqdqd", output_fddqdqd)
        

    #     for i in range(n):
    #         q_plus = np.copy(q)
    #         q_minus = np.copy(q) 

    #         q_plus[i] = q_plus[i] + h    
    #         q_minus[i] = q_minus[i] - h   

    #         iddq_p, iddqd_p = self.idsva(q_plus,qd,qdd)
    #         iddq_m, iddqd_m = self.idsva(q_minus,qd,qdd)
    #         _, _, fddt_plus = self.fdsva(q_plus,qd,qdd,tau,id_M, iddq_p, iddqd_p)   
    #         _, _, fddt_minus = self.fdsva(q_minus,qd,qdd,tau,id_M,iddq_m, iddqd_m)  

    #         temp = np.zeros((n,n))
    #         for j in range(n):
    #             temp[i][j] = (fddt_plus[i][j] - fddt_minus[i][j]) / (2*h)
            
    #         output_fddtdq.append(temp)
    #     # print("output_fddtdq", output_fddtdq)

    #     # fddqd_plus = input_fddqd.copy
    #     # fddqd_minus = input_fddqd.copy 

    #     # fddt_plus = input_fddt.copy
    #     # fddt_minus = input_fddt.copy


    #     fdsva_so = [output_fddqq,output_fddqd_dq,output_fddqdqd,output_fddtdq]

    #     return fdsva_so 

        
    
    def fdsva_parallel( self, q, qd, qdd, tau, id_M, iddq, iddqd):
       
        n = len(qd)
        fddq = np.zeros((n,n))
        fddqd = np.zeros((n,n))
        fddt = np.zeros((n,n))
        Minv = np.linalg.inv(id_M)

        # fddq = iddq
        # fddqd = iddqd
    
        # print("fddq\n", fddq)
        # print("fddq[0]\n", fddq[0])
        # print("fddq[1]\n", fddq[1])


        # print("Minv\n", Minv)
        fddq = Minv@iddq
        # print("fddq\n", fddq)
        fddq *= (-1)
        fddqd = (-1)*Minv@iddqd
        fddt = Minv 

        return fddq, fddqd, fddt
    
    def rotate(self, arr):
        pages = len(arr)
        rows = len(arr[0])
        cols = len(arr[0,0])

        # print(pages)
        # print(rows)
        # print(cols)
        # print(arr[2,3,5])
        for k in range(pages):
            for i in range(rows):
                for j in range(cols):
                    temp1 = arr[k,i,j]
                    # arr[k,i,j] = arr[j,i,k]
                    arr[j,i,k] = temp1
        # print(arr[5,3,2])
        return arr


    
    def fdsva_so(self, q, qd, qdd, tau, fd_dq, fd_dqd, df_dt, di2_dq, di2_dqd, di2_dq_dqd, dm_dq):
        fddqq = []
        fddqd_dq = []
        fddqdqd = []
        fddtdq = []
        Minv = self.minv(q)

        # temp = mdq@fddq
        # print(temp.shape)
        # print("temp", temp)
        # temp1 = self.rotate(temp)
        # print("temp1", temp1)
        # temp2 = np.rot90(temp, axes = (1,2))
        # print(temp2.shape)
        # print("temp2", temp2)


        fddqq = (-1)*Minv@(di2_dq + (dm_dq@fd_dq) + self.rotate(dm_dq@fd_dq))
        fddqd_dq = (-1)*Minv@(di2_dq_dqd + (dm_dq@fd_dqd))
        fddqdqd = (-1)*Minv@(di2_dqd)
        fddtdq = (-1)*Minv@(dm_dq@Minv)

        return fddqq,fddqd_dq, fddqdqd, fddtdq
    
    def fdsva_so_parallel(self, q, qd, qdd, tau, fddq, fddqd, fddt, iddqq, iddqdqd, iddqddqd, mdq):
        fddqq = []
        fddqd_dq = []
        fddqdqd = []
        fddtdq = []
        Minv = self.minv(q)

        # print("mdq", mdq)
        # print("fddq", fddq)
        temp = (mdq@fddq) 
        # print("temp", temp)
        # rot_temp = self.rotate(temp)
        # fddqq = (iddqq + temp + rot_temp)
        for i in range(7):
            for j in range(7):
                for k in range(7):
                    fddqq[i,j,k] = (iddqq[i,j,k] + temp[i,j,k])
        # print("fddqq", fddqq)


        temp = (mdq@fddqd)
        fddqd_dq = (iddqddqd + temp)

        fddqdqd = (iddqdqd)

        fddtdq = (mdq@Minv)

        
        fddqq = Minv@fddqq
        fddqd_dq = Minv@fddqd_dq
        fddqdqd = Minv@fddqdqd
        fddtdq = Minv@fddtdq

        
        fddqq *= (-1)
        fddqd_dq *= (-1)
        fddqdqd *= (-1)
        fddtdq *= (-1)

        return fddqq,fddqd_dq, fddqdqd, fddtdq

    def finite_diff_d2tau_dq_dqd(self, q, qd, tau):

        qdd = self.aba(q,qd,tau,-9.81)

        h = 0.01 
        n = len(qd)
        d2tau_dq_dqd = np.zeros((n,n,n))

        for i in range(n):
            qd_plus = np.copy(qd)
            qd_minus = np.copy(qd)

            qd_plus[i] = qd_plus[i] + h    
            qd_minus[i] = qd_minus[i] - h  
            iddq_p, iddqd_p = self.idsva(q,qd_plus,qdd)
            iddq_m, iddqd_m = self.idsva(q,qd_minus,qdd)

            # print("iddq_p", iddq_p)
            # print("iddq_m", iddq_m)

            # temp = np.copy(iddq_p) #np.zeros((n,m))
            # for j in range(n):
            #     temp[i][j] = (iddq_p[i][j] - iddq_m[i][j]) / (2*h)
            
            # d2tau_dq_dqd.append(temp)
            d2tau_dq_dqd[:,:,i] = np.squeeze((iddq_p - iddq_m) / (2*h))



        # print("d2tau_dq_dqd", d2tau_dq_dqd)
        return d2tau_dq_dqd

    def fdsva_so_2(self, q, qd, tau):
        df2_dq = []
        df2_dqd_dq = []
        df2_dqd = []
        df2_dt_dq = []
        df2_dq_dqd = []

        minv = self.minv(q, True)
        qdd = self.aba(q,qd,tau,-9.81)
        dtau_dq, dtau_dqd = self.idsva(q,qd,qdd)
        df_dq, df_dqd, df_du = self.fdsva(q,qd,qdd,tau,minv,dtau_dq,dtau_dqd)
        d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq = self.second_order_idsva_series(q,qd,qdd)

        dM_dq = [[[ 0,     -1.7669, -0.0375,  0.4489, -0.0122 , 0.0204, -0   ],
                [ 0,    -0.0973  ,1.2634 , 0.0692  ,0.03   ,-0.0239 ,-0   ],
                [ 0,     -1.279  ,-0.0169,  0.4199 ,-0.0117,  0.0279, -0    ],
                [ 0,      0.1688 ,-0.1428, -0.0036 ,-0.0111 , 0.0036,  0    ],
                [ 0,     -0.0154 , 0.0035, -0.0086 ,-0.0147, -0.0413, -0    ],
                [ 0,      0.0067 , 0.006,  -0.0039 ,-0.0514, -0.0061, -0    ],
                [ 0,      0.0049 , 0,     -0.0049  ,0.0006 , 0.0048 , 0    ]],

                [[ 0,    -0.0973 , 1.2634 , 0.0692 , 0.03 ,  -0.0239 ,-0   ],
                [ 0,      0,      0.2303 ,-2.0164 , 0.0797,  0.1434 , 0    ],
                [ 0,      0,      1.1739 , 0.0768 , 0.0267 ,-0.022  ,-0    ],
                [ 0,      0,     -0.1695 , 0.9887 ,-0.0397 ,-0.097  ,-0    ],
                [ 0,      0,      0.0112 ,-0.0288 , 0.0394 ,-0.0052 ,-0    ],
                [ 0,      0,     -0.005  ,-0.0493 ,-0.0061 , 0.0741 , 0    ],
                [ 0,      0,     -0.0048 ,-0.0003 ,-0.0026 , 0.0012  ,0    ]],

                [[ 0,    -1.279 , -0.0169 , 0.4199, -0.0117 , 0.0279, -0    ],
                [ 0,      0,      1.1739,  0.0768,  0.0267, -0.022,  -0    ],
                [ 0,      0,      0,      0.3501 ,-0.0101 , 0.0375 ,-0    ],
                [ 0,      0,      0,      0.0192 ,-0.0086 , 0,      0    ],
                [ 0,      0,      0,     -0.0136 ,-0.0193 ,-0.0474, -0    ],
                [ 0,      0,      0,     -0.0033 ,-0.0595 ,-0.0055 ,-0    ],
                [ 0,      0,      0,     -0.0047 , 0.0006 , 0.0046 , 0    ]],

                [[ 0,      0.1688 ,-0.1428, -0.0036, -0.0111,  0.0036,  0    ],
                [ 0,     0,     -0.1695  ,0.9887 ,-0.0397, -0.097 , -0    ],
                [ 0,      0,     0,      0.0192 ,-0.0086 , 0,      0    ],
                [ 0,      0,      0,      0,      0.0009 , 0.0525,  0    ],
                [ 0,      0,      0,      0,     -0.0282 , 0.0103, 0    ],
                [ 0,      0,      0,      0,      0.0137 ,-0.0254,-0   ],
                [ 0,      0,      0,      0,      0.0026 ,-0.0009 , 0    ]],

                [[ 0,     -0.0154 , 0.0035, -0.0086, -0.0147, -0.0413,-0    ],
                [ 0,      0,      0.0112, -0.0288,  0.0394, -0.0052, -0    ],
                [ 0,      0,      0,     -0.0136, -0.0193, -0.0474, -0    ],
                [ 0,      0,      0,      0,    -0.0282,  0.0103,  0    ],
                [ 0,      0,      0,      0,      0,     -0.0116 ,-0    ],
                [ 0,      0,      0,      0,      0,     -0,     -0    ],
                [ 0,      0,      0,      0,      0,      0.0027,  0    ]],

                [[ 0,      0.0067,  0.006,  -0.0039, -0.0514, -0.0061 ,-0    ],
                [ 0,      0,     -0.005,  -0.0493, -0.0061 , 0.0741,  0    ],
                [ 0,      0,      0,     -0.0033 ,-0.0595 ,-0.0055 ,-0    ],
                [ 0 ,     0,      0  ,    0,      0.0137 ,-0.0254, -0    ],
                [ 0,      0,      0,     0,     0,     -0,     -0    ],
                [ 0,      0,      0,      0,      0,      0,      0    ],
                [ 0,      0,      0,      0,      0,      0,      0    ]],

                [[ 0,      0.0049 , 0,     -0.0049 , 0.0006,  0.0048,  0    ],
                [ 0,      0,    -0.0048 ,-0.0003, -0.0026,  0.0012 , 0    ],
                [ 0,      0,      0,     -0.0047,  0.0006 , 0.0046,  0    ],
                [ 0,      0,      0,      0,      0.0026 ,-0.0009,  0    ],
                [ 0,      0,      0,      0,      0,      0.0027 , 0    ],
                [ 0,      0,      0,      0,      0,      0,      0    ],
                [ 0,      0,      0,      0,      0,     0,      0    ]]]

        d2tau_dq_dqd = self.finite_diff_d2tau_dq_dqd(q,qd,tau)
        # print("d2tau_dq_dqd", d2tau_dq_dqd)


        # print("d2tau_dq", d2tau_dq)
        # print("d2tau_dqd", d2tau_dqd)
        # print("d2tau_cross", d2tau_cross)
        # print("dM_dq", dM_dq)

        # print("mdq", dM_dq)
        # print(dM_dq[0]@df_dq)
        # print("fddq", df_dq)
        temp = (dM_dq@df_dq) 
        # print("dM_dq@df_dq", temp)
        rot_temp = self.rotate(temp)
        # print("rot_temp", rot_temp)
        df2_dq = (d2tau_dq + temp + rot_temp)
        # df2_dq = (d2tau_dq + temp)
        # print("d2tau_dq", d2tau_dq)
        # print("df2_dq", df2_dq)    

        temp = (dM_dq@df_dqd)
        # print("dM_dq@df_dqd", temp)
        df2_dqd_dq = (d2tau_cross + temp)
        # print("df2_dqd_dq", df2_dqd_dq)

        df2_dqd = (d2tau_dqd)
        # print("df2_dqd", df2_dqd)

        print("dM_dq", dM_dq)
        print("minv", minv)

        df2_dt_dq = (dM_dq@minv)
        print("df2_dt_dq", df2_dt_dq)

        df2_dq_dqd = self.rotate(dM_dq@df_dqd)
        df2_dq_dqd += d2tau_dq_dqd 


        df2_dq = minv@df2_dq
        df2_dqd_dq = minv@df2_dqd_dq
        df2_dqd = minv@df2_dqd
        df2_dt_dq = minv@df2_dt_dq
        df2_dq_dqd = minv@df2_dq_dqd

        df2_dq *= (-1)
        df2_dqd_dq *= (-1)
        df2_dqd *= (-1)
        df2_dt_dq *= (-1)
        df2_dq_dqd *= (-1)
        # print("df2_dq", df2_dq)
        # print("df2_dqd_dq", df2_dqd_dq)
        # print("df2_dqd", df2_dqd)
        print("df2_dt_dq", df2_dt_dq)



        # fddqq = (-1)*minv@(d2tau_dq + (dM_dq@df_dq) + self.rotate(dM_dq@df_dq))
        # fddqd_dq = (-1)*minv@(d2tau_cross + (dM_dq@df_dqd))
        # fddqdqd = (-1)*minv@(d2tau_dqd)
        # fddtdq = (-1)*minv@(dM_dq@minv)


        return df2_dq, df2_dqd_dq, df2_dqd, df2_dt_dq, df2_dq_dqd



    
