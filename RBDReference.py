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
        CRBA calculates the joint space inertia matrix H to represent the composite inertia
        This is used in the fundamental motion equation H qdd + C = Tau
        Forward dynamics roughly calculates acceleration as H_inv ( Tau - C); analytic inverse - benchmark against Pinocchio
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
                print("U", U)
                d[ind] = S @ U[:, ind]
                print("d",d)
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
                       
        n = len(qd)
        
        C = self.rnea(q, qd, qdd = None, GRAVITY = -9.81)[0]
        
        IC = copy.deepcopy(self.robot.get_Imats_dict_by_id())# composite inertia calculation

        for ind in range(n-1,-1,-1):
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

        sub=np.subtract(tau,C) 
        return H
    
    def crba_parallel( self, q, qd, tau):
        n = len(qd)
        C = self.rnea(q, qd, qdd=None, GRAVITY=-9.81)[0]

        IC = copy.deepcopy(self.robot.get_Imats_dict_by_id()) 
        H = np.zeros((n, n))

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
            H[ind, ind] = np.matmul(S, fh)
                    
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
                H[ind, j] = np.matmul(S.T, fh)
                H[j, ind] = H[ind, j]
                            
        return H


    
