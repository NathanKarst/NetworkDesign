import viscosity
import skimming
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.io
import pickle
import random


class Network:
    def __init__(self,params):
        self.seed = None
        for key in params:
            setattr(self, key, params[key])
            
            
        if not self.seed:
            self.seed = 1234

        self.params = params
        
        self.set_adj_inc() 
        
        self.p = np.array((self.nNodes)*[np.nan])
        self.q = np.array((self.nVessels)*[np.nan])
        self.h = np.array(self.nVessels*[np.nan])
        self.equilibria = []
        

    #########################
    ##### BASIC METHODS #####
    #########################
    
    def set_parameter(self,paramName,paramValue,index=None):
        if index == None:
            self.__setattr__(paramName,paramValue)
        else:
            tmp = getattr(self,paramName)
            tmp[index] = paramValue
            self.__setattr__(paramName,tmp)        

    def set_adj_inc(self):
        n_v = len(self.v)
        n_e = len(self.e)
        adj = np.zeros((n_v,n_v))
        inc = np.zeros((n_v,n_e))

        for i,edge in enumerate(self.e):
            n0 = min(edge)
            n1 = max(edge)

            adj[n0,n1] = 1
            adj[n1,n0] = 1

            # directed edge from lower index to higher index
            inc[n0,i] = -1
            inc[n1,i] = 1
            
        self.adj = adj
        self.inc = inc
        
        self.nNodes = int(adj.shape[0])
        self.nVessels = int(inc.shape[1])

        self.interiorNodes = np.where(np.sum(np.abs(inc),axis=1) == 3)[0]
        self.exteriorNodes = np.where(np.sum(np.abs(inc),axis=1) == 1)[0]
        self.exteriorFlows = np.where(np.sum(np.abs(inc[self.exteriorNodes,:]),axis=0) == 1)[0]
        self.interiorFlows = np.where(np.sum(np.abs(inc[self.exteriorNodes,:]),axis=0) != 1)[0]
        
        self.nInteriorNodes = len(self.interiorNodes)
        self.nExteriorNodes = len(self.exteriorNodes)
        self.nExteriorFlows = len(self.exteriorFlows) 
        self.nInteriorFlows = len(self.interiorFlows)

        if self.bc_type == 'flow':
            np.random.seed(self.seed)
            self.k = self.inc.shape[1] - self.inc.shape[0] + 1
            self.q0 = np.linalg.lstsq(self.inc,self.pq_bcs,rcond=None)[0]
            
            V = sp.linalg.svd(self.inc)[2].T
            self.K = V[:,-self.k:]    
    
    def set_flow_boundary_conditions(self,kind,inlets=None,seed=2024):
        q_ext = np.zeros(self.nNodes)
        if not inlets:
            print('Must specify at least one inlet node')
            return
        for inlet in inlets:
            if inlet not in self.exteriorNodes:
                print(f'Node {inlet} is not exterior.')
                return
        
        outlets = list(set(self.exteriorNodes) - set(inlets))
        
        if kind == 'equal':
            q_ext[inlets] = -1/len(inlets)
            q_ext[outlets] = 1/len(outlets)
        elif kind == 'random':
            np.random.seed(seed)
            tmp = np.random.random(len(inlets))
            q_ext[inlets] = -tmp/np.sum(tmp)
            
            tmp = np.random.random(len(outlets))
            q_ext[outlets] = tmp/np.sum(tmp)        
        else:
            print('Kind not supported')
            return
        self.pq_bcs = q_ext
        self.bc_type = 'flow'
        self.set_adj_inc()
    
    def set_hematocrit_boundary_conditions(self,kind,hmin=0,hmax=0.45,seed=2024):
        h_ext = np.zeros(self.nVessels)
        if kind == 'equal':
            h_ext[self.exteriorFlows] = hmin
        elif kind == 'random':
            h_ext[self.exteriorFlows] = np.random.uniform(hmin,hmax,self.exteriorFlows)
        else:
            print('Kind not supported')
            return
        self.h_bcs = h_ext        
        
    ###############################
    ##### EQUILIBRIUM METHODS #####
    ###############################        
        
    def set_bh(self,h):
        poiseuille = np.block([self.inc.T, np.zeros((self.nVessels,self.nVessels))])
        r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        for i in range(self.nVessels):
            poiseuille[i,self.nNodes+i] = r[i]    
        
        interior_flow_balance = np.block([np.zeros((len(self.interiorNodes),self.nNodes)),self.inc[self.interiorNodes,:]])
        boundary_conditions = np.zeros((len(self.exteriorNodes),self.nNodes + self.nVessels))
        if self.bc_type == 'pressure':
            for i,node in enumerate(self.exteriorNodes):
                boundary_conditions[i,node] = 1
            tail = self.pq_bcs[self.exteriorNodes]
        elif self.bc_type == 'flow':
            sign_correction = np.ones(self.nExteriorNodes)
            for i,node in enumerate(self.exteriorNodes[:-1]):
                flow = np.where(self.inc[node,:])[0]
                boundary_conditions[i,self.nNodes+flow] = 1
                sign_correction[i] = self.inc[node,:].sum()
         
            tail = self.pq_bcs[self.exteriorNodes]*sign_correction
            
            boundary_conditions[len(self.exteriorNodes)-1,self.exteriorNodes[0]] = 1            
            tail[-1] = 1
            
        self.bh = np.concatenate([poiseuille, interior_flow_balance, boundary_conditions],axis=0)
        self.bh_rhs = np.concatenate((np.zeros(self.nNodes + self.nVessels - len(self.exteriorNodes)), tail))                 
        
    def set_cq(self):
        rhs = np.zeros(self.nVessels)
        
        X = self.inc@np.diag(np.sign(self.q))
        exterior = np.array((np.abs(self.inc).sum(axis=1) == 1)).flatten()    
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_in = np.array(X.sum(axis=1) == 1).flatten()
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        
        CQ = self.inc[self.interiorNodes,:]@np.diag(self.q)
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        for node in div:
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
            row = np.zeros(self.nVessels)
            row[idx_in] = -skimming.skimming_kj(np.abs(self.q[idx_out]/self.q[idx_in]),self.pPlasma)[0]
            row[idx_out] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))
        
        inlets =  np.where(np.bitwise_and(exterior,net_out))[0]
        for i,inlet in enumerate(inlets):
            row = np.zeros(self.nVessels)
            inlet_vessel = np.where(self.inc[inlet])[0][0]            
            
            row[inlet_vessel] = 1
            CQ = np.concatenate((CQ,row.reshape(1,-1)))              
            rhs[-(len(inlets)-i)] = self.h_bcs[inlet_vessel]

        self.cq = CQ
        self.cq_rhs = rhs 
    
    def set_pq(self,h):
        self.set_bh(h)
        pq = np.linalg.solve(self.bh,self.bh_rhs)
        self.p = pq[:self.nNodes]
        self.q = pq[self.nNodes:]
    
    def set_state(self,h):
        self.h = h
        self.set_pq(h)
        # self.set_bh(h) # not necessary, as B(H) is set in self.set_pq
        self.set_cq()
        self.r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        
    def find_equilibria(self,n,tol=1e-4,verbose=False,fragile=False):
        from ipywidgets import IntProgress
        from IPython.display import display
        f = IntProgress(min=0, max=n);
        display(f)
        
        if self.seed: np.random.seed(self.seed)
        self.equilibria = []
        i = 0 
        while i < n:
            h = np.random.random(self.nVessels)
            
            if fragile:
                outcome = fsolve(self.equilibrium_relation, h)
            else:
                try: 
                    outcome = fsolve(self.equilibrium_relation, h)
                except:
                    continue
            residual = np.linalg.norm(self.equilibrium_relation(outcome))/len(outcome)
            if residual > tol:
                continue
        
            self.set_state(outcome)
             
            if verbose:
                print(f'{i+1}: |F(x*)| = {np.round(residual,6)}')
            if i == 0:
                self.equilibria = self.h.reshape(-1,1)
            else:
                self.equilibria = np.concatenate((self.equilibria,self.h.reshape(-1,1)),axis=1)
            i += 1  
            f.value += 1

            
            
    ################################
    ##### CONTINUATION METHODS #####
    ################################
        
    def set_jacobian(self,h):
        self._curr_h = h

        self.set_state(h)
        
        B = self.bh.copy()
        C = self.cq.copy()
        D = np.zeros((B.shape[0],C.shape[1]))
        E = np.zeros((C.shape[0],B.shape[1]))
        
        D[:self.nVessels,:self.nVessels] = np.diag(self.delta*self.q*self.r)
       
        X = self.inc@np.diag(np.sign(self.q))
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        n_div = len(div)
        
        dsdh = np.zeros((n_div,self.nVessels))
        dsdq = np.zeros((n_div,self.nVessels))
        for i,node in enumerate(div):
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
           
            q_in = self.q[idx_in]
            q_out = self.q[idx_out]
        
            dsdh[i,idx_in] = -skimming.skimming_kj(np.abs(q_out/q_in),self.pPlasma)[0]
            dsdh[i,idx_out] = 1
            
            dfdq = skimming.skimming_kj_dq(np.abs(q_out/q_in),self.pPlasma)[0]
            
            dsdq[i,idx_in] = self.h[idx_in]*dfdq*np.abs(q_out)/(q_in**2)*np.sign(q_in)
            dsdq[i,idx_out] = -self.h[idx_in]*dfdq/np.abs(q_in)*np.sign(q_out)

        C[self.nInteriorNodes:self.nInteriorNodes + n_div, :] = dsdh            
            
            
        E[:self.nInteriorNodes,self.nNodes:] = self.inc[self.interiorNodes,:]@np.diag(h)            
        E[self.nInteriorNodes:self.nInteriorNodes + n_div, self.nNodes:] = dsdq
        
        self.jacobian = np.block([[B,D],[E,C]])
    
    def get_jacobian(self,h):
        self.set_jacobian(h)
        return self.jacobian
    
    def get_jacobian_from_full_state(self,x):
        p = x[:self.nNodes]
        q = x[self.nNodes:self.nNodes+self.nVessels]
        h = x[-self.nVessels:]
        
        poiseuille = np.block([self.inc.T, np.zeros((self.nVessels,self.nVessels))])
        r = self.l/self.d**4*viscosity.viscosity_arrhenius(h,self.delta)
        for i in range(self.nVessels):
            poiseuille[i,self.nNodes+i] = r[i]    
        
        interior_flow_balance = np.block([np.zeros((len(self.interiorNodes),self.nNodes)),self.inc[self.interiorNodes,:]])
        boundary_conditions = np.zeros((len(self.exteriorNodes),self.nVessels + self.nNodes))
        if self.bc_type == 'pressure':
            for i,node in enumerate(self.exteriorNodes):
                boundary_conditions[i,node] = 1
        elif self.bc_type == 'flow':
            for i,node in enumerate(self.exteriorNodes[:-1]):
                flow = np.where(self.inc[node,:])[0]
                boundary_conditions[i,self.nNodes+flow] = 1
            
            boundary_conditions[len(self.exteriorNodes)-1,self.exteriorNodes[0]] = 1  
            
        B = np.concatenate([poiseuille, interior_flow_balance, boundary_conditions],axis=0)
        
        X = self.inc@np.diag(np.sign(q))
        exterior = np.array((np.abs(self.inc).sum(axis=1) == 1)).flatten()    
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        net_in = np.array(X.sum(axis=1) == 1).flatten()    
        
        C = self.inc[self.interiorNodes,:]@np.diag(q)
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        for node in div:
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
            row = np.zeros(self.nVessels)
            row[idx_in] = -skimming.skimming_kj(np.abs(q[idx_out]/q[idx_in]),self.pPlasma)[0]
            row[idx_out] = 1
            C = np.concatenate((C,row.reshape(1,-1)))  
        
        inlets =  np.where(np.bitwise_and(exterior,net_out))[0]
        for i,inlet in enumerate(inlets):
            row = np.zeros(self.nVessels)
            inlet_vessel = np.where(self.inc[inlet])[0][0]            
            
            row[inlet_vessel] = 1
            C = np.concatenate((C,row.reshape(1,-1)))                 
        
        D = np.zeros((B.shape[0],C.shape[1]))
        E = np.zeros((C.shape[0],B.shape[1]))
        
        D[:self.nVessels,:self.nVessels] = np.diag(self.delta*q*r)
       
        X = self.inc@np.diag(np.sign(q))
        interior = np.array((np.abs(self.inc).sum(axis=1) == 3)).flatten()    
        net_out = np.array(X.sum(axis=1) == -1).flatten()
        
        div = np.where(np.bitwise_and(interior,net_out))[0]
        n_div = len(div)
        
        dsdh = np.zeros((n_div,self.nVessels))
        dsdq = np.zeros((n_div,self.nVessels))
        for i,node in enumerate(div):
            idx_in = np.where(X[node,:] > 0)[0][0]
            idx_out = np.min(np.where(X[node,:] < 0)[0])
           
            q_in = q[idx_in]
            q_out = q[idx_out]
        
            dsdh[i,idx_in] = -skimming.skimming_kj(np.abs(q_out/q_in),self.pPlasma)[0]
            dsdh[i,idx_out] = 1
            
            dfdq = skimming.skimming_kj_dq(np.abs(q_out/q_in),self.pPlasma)[0]
            
            dsdq[i,idx_in] = h[idx_in]*dfdq*np.abs(q_out)/(q_in**2)*np.sign(q_in)
            dsdq[i,idx_out] = -h[idx_in]*dfdq/np.abs(q_in)*np.sign(q_out)

        C[self.nInteriorNodes:self.nInteriorNodes + n_div, :] = dsdh            
            
            
        E[:self.nInteriorNodes,self.nNodes:] = self.inc[self.interiorNodes,:]@np.diag(h)            
        E[self.nInteriorNodes:self.nInteriorNodes + n_div, self.nNodes:] = dsdq
        
        J = np.block([[B,D],[E,C]])
        
        return J
    
    def get_hessian_tensor(self,x):
        d = 1e-3
        
        H = []
        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += -d
            x2[i] += d
            
            J1 = self.get_jacobian_from_full_state(x1)
            J2 = self.get_jacobian_from_full_state(x2)
            DJ = (J2 - J1)/2/d
            H.append(np.expand_dims(DJ,2))
            
        H = np.concatenate(H,axis=2)
        
        return H
    
    
#     def set_hessian(self,h):
#         h_orig = h.copy()
        
#         self.set_state(h)
        
#         x = np.concatenate([self.p, self.q, self.h])
        
        
        
        
        
        
    def equilibrium_relation(self,h):
        self.set_state(h)
        
        return h - np.linalg.solve(self.cq,self.cq_rhs)
    
    def fold_relation(self,h):
        self.set_jacobian(h)
        
        return(np.linalg.det(self.jacobian))
    
    def isola_center_relation(self,h):
        self.set_jacobian(h)
        
        L,V = sp.linalg.eig(self.jacobian.T)
        
        i = np.argmin(np.abs(L))
        psi = V[:,i]

        return np.real(psi[self.nVessels + self.nInteriorNodes])*10
        

    ##############################
    ##### EIGENVALUE METHODS #####
    ##############################
    
#     def _dhdqAll(self,s,w):
#         self._dhdq = np.zeros(len(self.q)*len(self.q)*s.shape[0]*s.shape[1])*1j
#         self._dhdq = self._dhdq.reshape((len(self.q),len(self.q),s.shape[0],s.shape[1]))
        
#         sorted_nodes = np.argsort(self.p)[::-1]

#         inlets = []
#         for node in sorted_nodes:
#             connected_edges = np.where(self.inc[node,:] != 0)[0]

#             in_flows = []
#             out_flows = []

#             for edge in connected_edges:
#                 if self.inc[node,edge] == 1 and self.q[edge] > 0:
#                     in_flows.append(edge)
#                 elif self.inc[node,edge] == -1 and self.q[edge] < 0:
#                     in_flows.append(edge)
#                 else:
#                     out_flows.append(edge)
# #             print(f'node = {node}')
# #             print(f'in = {in_flows}')
# #             print(f'out = {out_flows}')
# #             print()
#             if len(in_flows) == 0 and len(out_flows) == 1: # then inlet
#                 continue
#                 # print(f'Inlet at {node}')
#                 # edge = connected_edges[0]
#                 # self._pert[edge] = 1
#                 # inlets.append(node)
#                 # self._dhdq[edge,edge,:,:] = self.h_bcs[edge]
                
#             elif len(in_flows) == 2 and len(out_flows) == 1: # then converger
# #                 print(f'Converger at {node}')
#                 self._dhdq += self._dhdqCon(*in_flows,*out_flows)
#             elif len(in_flows) == 1 and len(out_flows) == 2: # then diverger
# #                 print(f'Diverger at {node}')                
#                 self._dhdq += self._dhdqDiv(*in_flows,*out_flows)
                
# #         for inlet in inlets:
# #             print(inlet)
# #             self._dhdq[:,inlet,:,:] = 0
# #             self._pert[inlet] = 0
            
# #         print(self._dhdq[:,0,0,0])            

#     def _dhdqCon(self,idxa,idxb,idxo):
#         out = np.zeros(np.shape(self._dhdq))*1j
#         qa = self.q[idxa]
#         ha = self.h[idxa]
#         qb = self.q[idxb]
#         hb = self.h[idxb]
#         qo = self.q[idxo]

#         for k in range(len(self.q)):
#             if k == idxa:
#                 out[idxo,k,:,:] += (self._dhdq[idxa,k,:,:]*self._pert[idxa,:,:]*abs(qa) + ha*np.sign(qa) + self._dhdq[idxb,k,:,:]*self._pert[idxb,:,:]*abs(qb))/abs(qo)
#             elif k == idxb:
#                 out[idxo,k,:,:] += (self._dhdq[idxb,k,:,:]*self._pert[idxb,:,:]*abs(qb) + hb*np.sign(qb) + self._dhdq[idxa,k,:,:]*self._pert[idxa,:,:]*abs(qa))/abs(qo)
#             elif k == idxo:
#                 out[idxo,k,:,:] += -(ha*abs(qa) + hb*abs(qb))/qo**2*np.sign(qo)
#             else:
# #                 continue
#                 out[idxo,k,:,:] += (self._dhdq[idxa,k,:,:]*self._pert[idxa,:,:]*abs(qa) + self._dhdq[idxb,k,:,:]*self._pert[idxb,:,:]*abs(qb))/abs(qo)

#         return out    
    
    
#     def _dhdqDiv(self,idxf,idxa,idxb):
#         out = np.zeros(np.shape(self._dhdq))*1j
        
#         hf = self.h[idxf]
#         qf = self.q[idxf]
#         qa = self.q[idxa]

#         f,g = skimming.skimming_kj(abs(qa/qf),self.pPlasma)
#         dfdq, dgdq = skimming.skimming_kj_dq(abs(qa/qf),self.pPlasma)
#         dfdh, dgdh = skimming.skimming_kj_dh(abs(qa/qf),self.pPlasma)

#         for k in range(len(self.q)):
#             if k == idxf:
#                 out[idxa,k,:,:] += self._dhdq[idxf,k,:,:]*self._pert[idxf,:,:]*(f + hf*dfdh) - hf*dfdq*abs(qa)/qf**2*np.sign(qf)
#                 out[idxb,k,:,:] += self._dhdq[idxf,k,:,:]*self._pert[idxf,:,:]*(g + hf*dgdh) - hf*dgdq*abs(qa)/qf**2*np.sign(qf)
#             elif k == idxa:
#                 out[idxa,k,:,:] += hf*dfdq*np.sign(qa)/abs(qf)
#                 out[idxb,k,:,:] += hf*dgdq*np.sign(qa)/abs(qf)
# #             elif k == idxb: # handlded in previous block?
# #                 continue 
#             else:
# #                 continue
#                 out[idxa,k,:,:] += self._dhdq[idxf,k,:,:]*self._pert[idxf,:,:]*(f + hf*dfdh)
#                 out[idxb,k,:,:] += self._dhdq[idxf,k,:,:]*self._pert[idxf,:,:]*(g + hf*dgdh)

#         return out

    
    def relation_hopf(self,s,w):
        v = np.pi*(self.d/2)**2*self.l
        tau = v/np.abs(self.q)/np.sum(v[self.interiorFlows])
        lamb = s + np.array(1j)*w
        
        def tensor_const_over_sw(mat):
            return np.tensordot(np.ones((s.shape)),mat,axes=0)
        
        drdm = tensor_const_over_sw(np.diag(self.l/self.d**4))
        dmdh = tensor_const_over_sw(np.diag(viscosity.viscosity_arrhenius_deriv(self.h,self.delta)))
        q = tensor_const_over_sw(np.diag(self.q))
        r = tensor_const_over_sw(np.diag(self.r))
        
        eye = tensor_const_over_sw(np.eye(len(self.q)))
        exp = np.exp(np.tensordot(-lamb,np.diag(tau),axes=0))*eye
        
        pert = np.tensordot(1/lamb,np.diag(1/tau),axes=0)*(eye - exp)
        
        dhdq = np.zeros(q.shape)*1j
        
        sorted_nodes = np.argsort(self.p)[::-1]

        inlets = []
        for node in sorted_nodes:
            connected_edges = np.where(self.inc[node,:] != 0)[0]

            in_flows = []
            out_flows = []

            for edge in connected_edges:
                if self.inc[node,edge] == 1 and self.q[edge] > 0:
                    in_flows.append(edge)
                elif self.inc[node,edge] == -1 and self.q[edge] < 0:
                    in_flows.append(edge)
                else:
                    out_flows.append(edge)
                   
            if len(in_flows) == 2 and len(out_flows) == 1: # then converger
                idxa = in_flows[0]
                idxb = in_flows[1]
                idxo = out_flows[0]
               
                qa = self.q[idxa]
                ha = self.h[idxa]
                qb = self.q[idxb]
                hb = self.h[idxb]
                qo = self.q[idxo]
                
                for k in range(self.nVessels):
                    if k == idxa:
                        tmp = dhdq[:,:,idxa,idxa]*exp[:,:,idxa,idxa]*abs(qa) 
                        tmp += ha*np.sign(qa)
                        tmp += dhdq[:,:,idxb,idxa]*exp[:,:,idxb,idxb]*abs(qb)
                        dhdq[:,:,idxo,idxa] += tmp/abs(qo)
                    elif k == idxb:
                        tmp = dhdq[:,:,idxb,idxb]*exp[:,:,idxb,idxb]*abs(qb)
                        tmp += hb*np.sign(qb)
                        tmp += dhdq[:,:,idxa,idxb]*exp[:,:,idxa,idxa]*abs(qa)
                        dhdq[:,:,idxo,idxb] += tmp/abs(qo)
                    elif k == idxo:
                        dhdq[:,:,idxo,idxo] += -(ha*abs(qa) + hb*abs(qb))/qo**2*np.sign(qo)
                    else:
                        tmp = dhdq[:,:,idxa,k]*exp[:,:,idxa,idxa]*abs(qa) 
                        tmp += dhdq[:,:,idxb,k]*exp[:,:,idxb,idxb]*abs(qb)
                        dhdq[:,:,idxo,k] += tmp/abs(qo)
              
            elif len(in_flows) == 1 and len(out_flows) == 2: # then diverger
                idxf = in_flows[0]
                idxa = out_flows[0]
                idxb = out_flows[1]
                
                hf = self.h[idxf]
                qf = self.q[idxf]
                qa = self.q[idxa]

                f,g = skimming.skimming_kj(abs(qa/qf),self.pPlasma)
                dfdq, dgdq = skimming.skimming_kj_dq(abs(qa/qf),self.pPlasma)
                dfdh, dgdh = skimming.skimming_kj_dh(abs(qa/qf),self.pPlasma)
                
                for k in range(self.nVessels):
                    if k == idxa:
                        dhdq[:,:,idxa,idxa] += hf*dfdq*np.sign(qa)/abs(qf)
                        dhdq[:,:,idxb,idxa] += hf*dgdq*np.sign(qa)/abs(qf)                        
                    elif k == idxf:
                        tmp = dhdq[:,:,idxf,idxf]*exp[:,:,idxf,idxf]*(f + hf*dfdh)
                        tmp += -hf*dfdq*abs(qa)/qf**2*np.sign(qf)
                        dhdq[:,:,idxa,idxf] += tmp

                        tmp = dhdq[:,:,idxf,idxf]*exp[:,:,idxf,idxf]*(g + hf*dgdh)
                        tmp += -hf*dgdq*abs(qa)/qf**2*np.sign(qf)
                        dhdq[:,:,idxb,idxf] += tmp
                    else:                
                        dhdq[:,:,idxa,k] += dhdq[:,:,idxf,k]*exp[:,:,idxf,idxf]*(f + hf*dfdh)
                        dhdq[:,:,idxb,k] += dhdq[:,:,idxf,k]*exp[:,:,idxf,idxf]*(g + hf*dgdh)
                        
        dkdp = tensor_const_over_sw(self.K.T)
        dqda = tensor_const_over_sw(self.K)
        
        char_mat = dkdp@(r + q@drdm@dmdh@pert@dhdq)@dqda
        char_eqn = np.linalg.det(char_mat)
        return char_eqn
    
    
    
#     def relation_hopf_safe(self,s,w):
#         v = np.pi*(self.d/2)**2*self.l
#         tau = v/np.abs(self.q)/np.sum(v[self.interiorFlows]) ## scaling issues compared to three-node paper
#         lamb = s + np.array(1j)*w
#         self._pert = np.array([np.exp(-lamb*t) for t in tau])
        
#         dRdEta = self.l/self.d**4
#         dEtadH = viscosity.viscosity_arrhenius_deriv(self.h,self.delta)
#         self._dhdqAll(s,w)
#         dHdQ = self._dhdq
        
#         pert = np.array([(1 - np.exp(-lamb*t))/(lamb*t) for t in tau])
#         pert = np.tensordot(pert[:,None,:,:],np.ones((1,self.nVessels)),axes=([1],[0]))
#         pert = np.transpose(pert,[0,3,1,2])

#         mat_constant_over_cols = lambda vec: vec.reshape(-1,1)@np.ones((1,self.nVessels))
#         Q = mat_constant_over_cols(self.q)
#         dRdEta = mat_constant_over_cols(dRdEta)
#         dEtadH = mat_constant_over_cols(dEtadH)

#         def tensor_const_over_sw(mat):
#             matFlat = (mat.shape[0]*mat.shape[1],1)
#             swFlat = (1,s.shape[0]*s.shape[1])
#             newShape = (mat.shape[0],mat.shape[1],s.shape[0],s.shape[1])
#             return (mat.reshape(matFlat)@np.ones(swFlat)).reshape(newShape)

#         Q = tensor_const_over_sw(Q)
#         dRdEta = tensor_const_over_sw(dRdEta)
#         dEtadH = tensor_const_over_sw(dEtadH)

#         dPdQ = np.zeros((len(self.q),len(self.q),s.shape[0],s.shape[1]))*1j
#         dPdQ += tensor_const_over_sw(np.diag(self.r))
#         dPdQ += Q*dRdEta*dEtadH*dHdQ*pert
        
#         dReldP = self.K.T 
#         dQdQSmall = self.K
# #         print(dReldP)
# #         print(dQdQSmall)
    
# #         print(f'shape(dPdQ): {dPdQ.shape}')
# #         print(dPdQ[:,:,0,0])
# #         print(f'shape(dReldP): {dReldP.shape}')  

#         out = np.tensordot(dReldP,dPdQ,axes=([1],[0]))
        
# #         print(f'shape(dReldP dot dPdQ): {out.shape}')  
        
#         out = np.tensordot(out,dQdQSmall,axes=([1],[0]))
        
# #         out = dReldP@dPdQ@dQdQSmall
# #         print(f'shape(dReldP dot dPdQ dot dQda): {out.shape}')          
        
#         out = np.transpose(out,[1,2,0,3])
# #         print(out.shape)
#         return np.linalg.det(out)    
    
    def computeSW(self,sMin,sMax,wMin,wMax,nGrid = 10):
        s = np.linspace(sMin,sMax,nGrid+1)
        w = np.linspace(wMin,wMax,nGrid)
        s,w = np.meshgrid(s,w)
        hopf = self.relation_hopf(s,w)
        return hopf,s,w

    def plotSW(self,hopf,s,w):
        plt.contour(s,w,np.real(hopf),[0],colors=['r'])
        plt.contour(s,w,np.imag(hopf),[0],colors=['b'])
        plt.plot([0,0],[np.min(w),np.max(w)],'k')
        plt.xlabel(r'$\sigma$')
        plt.ylabel(r'$\omega$')    
    
    
        
    #########################
    ##### ISOLA METHODS #####
    #########################
    
    def isola_stuff(self,x):
        J = self.get_jacobian_from_full_state(x)
        
        test_lambda = np.zeros(J.shape[1])
        test_lambda[0] = 1.0

        test_tau = np.zeros(J.shape[1])
        test_tau[self.nNodes + self.nVessels + 1] = 1.0

        for i in range(J.shape[0]):
            if (J[i,:] == test_lambda).all():
                idx_lambda = i
            elif (J[i,:] == test_tau).all():   
                idx_tau = i
        
        L,V = sp.linalg.eig(J)
        i = np.argmin(np.abs(L))
        phi = V[:,i]
        
        L,V = sp.linalg.eig(J.T)
        i = np.argmin(np.abs(L))
        psi = V[:,i]       
        
        F_lambda = np.zeros(J.shape[0])
        F_lambda[idx_lambda] = -1
        
        F_tau = np.zeros(J.shape[0])
        F_tau[idx_tau] = -1
       
        
        K = np.concatenate((J,100*phi.reshape((-1,1)).T),axis=0)          
        Z1 = np.linalg.lstsq(K,np.append(-F_lambda,0))[0]
        
        K = np.concatenate((J,psi.reshape((-1,1)).T),axis=0)          
        Z2 = np.linalg.lstsq(K,np.append(-F_tau,0))[0]
                             
        print(Z1)
        print(Z2)
        print()
        
        print(np.dot(Z1,phi))
        print(np.dot(Z2,phi))
                            
        print(J@Z1)
        print(J@Z2)
        
        H = self.get_hessian_tensor(x)
        
        a = np.dot(psi,H@phi@phi)
        b = np.dot(psi,H@phi@Z1)
        c = np.dot(psi,H@Z1@Z1)
        d = np.dot(psi,F_tau)
        
        
        tau_1 = 0
        tau_2 = np.sign(-a*d)
                
#         acot = lambda z: 1j/2*(np.log((z-1j)/z) - np.log((z+1j)/z))
#         beta = 1/2*acot((a-c)/2/b)
        
        
        return (H,phi)
       
            
    ###################################################        
    ######## TRANSIT TIME DISTRIBUTION METHODS ########
    ###################################################    
            
    def directed_adj_dict(self):
        q = self.pq[self.nNodes:]
        A = {}
        for i in range(len(q)):
            v0 = min(self.e[i])
            v1 = max(self.e[i])
            if q[i] > 0:
                A[v0] = A.get(v0,[]) + [v1]
            else:
                A[v1] = A.get(v1,[]) + [v0]

        for key,item in A.items():
            A[key] = set(item)

        for i in set(range(int(self.adj.shape[0]))) - set(A.keys()):
            A[i] = set([])

        self._adj_dict = A
     
    def get_paths_by_node_from_inlet(self,inlet):
        stack = [(inlet,[inlet])]
        paths = []
        while stack:
            (vertex, path) = stack.pop()

            if len(self._adj_dict[vertex]) == 0:
                paths.append(path)
            for next in self._adj_dict[vertex] - set(path):
                stack.append((next, path + [next]))            
    
        return paths
       
    def get_paths_by_node(self):
        self.directed_adj_dict()
        paths = []
        for i in self.exteriorNodes:
            new = self.get_paths_by_node_from_inlet(i)
            if len(new[0]) == 1:
                continue
            paths += new   
        self._paths_by_node = paths
    
    def get_paths_by_edge(self):
        self.get_paths_by_node()
        self._paths_by_edge = [[np.abs(self.inc)[[path[i],path[i+1]],:].sum(axis=0).argmax() for i in range(len(path)-1)] for path in self._paths_by_node]        

    def compute_conditional_probabilities_downstream(self):
        self._rbc = self.h*np.abs(self.q)   

        rbc_normalizer = np.zeros(self._rbc.shape)

        X = self.inc@np.diag(np.sign(self.q))
        for node in range(self.nNodes):
            row = X[node,:]
            if np.abs(row).sum() == 1:
                if row.sum() == -1:
                    vessel = np.where(row)[0][0]
                    rbc_normalizer[vessel] = np.sum(self._rbc[self.exteriorFlows])/2
            else:
                if row.sum() == -1:
                    inflow = np.where(row == 1)[0][0]
                    outflow_0 = np.where(row == -1)[0][0]
                    outflow_1 = np.where(row == -1)[0][1]                    
                    rbc_normalizer[outflow_0] = self._rbc[inflow]
                    rbc_normalizer[outflow_1] = self._rbc[inflow]                    
                elif row.sum() == 1:
                    outflow = np.where(row == -1)[0][0]
                    rbc_normalizer[outflow] = self._rbc[outflow]
        
        self._cond_prob = self._rbc/rbc_normalizer        
        
    def compute_ttd(self,verbose=False):
        self.compute_conditional_probabilities_downstream()
        if verbose:
            print(f'Cond prob: {self._cond_prob}')
        
        self.get_paths_by_edge()
        
        probs = []
        for path in self._paths_by_edge:
            probs.append(np.product(self._cond_prob[path]))

        if verbose: 
            for i,prob in enumerate(probs):
                print(f'P(RBC -> {self._paths_by_edge[i]}) \t= {np.round(prob,6)}')
            print(f'Check sum of total probability : {np.sum(probs)}')

        vol = np.pi*(self.d/2)**2*self.l
        tau = vol/np.abs(self.q)

        delays = []
        for path in self._paths_by_edge:
            delays.append(np.sum(tau[path]))

        if verbose: 
            for i,delay in enumerate(delays):
                print(f'{self._paths_by_edge[i]} :\t {delay}')
                
        ttd = TransitTimeDistribution(self._paths_by_edge, delays, probs)
#         if np.abs(np.sum(ttd.probs) - 1) > 1e-3:
#             print('Warning: Cumul. prob. of candidate TTD is not equal to 1!')
        self.ttd = ttd

    ###################################
    ###### VISUALIZATION METHODS ######
    ###################################    
    
    def plot(self,width=[],colors=[],directions=[],annotate=False,ms=10,x_min=0,x_max=1,y_min=0,y_max=1,annot_offset_x=[],annot_offset_y=[],ax=None,head_width=0.2,publication=False,line_style='-'):
        if len(colors) == 0: colors = len(self.e)*['k']
        if len(width) == 0: width = len(self.e)*[1]
        if len(annot_offset_x) == 0: annot_offset_x = len(self.e)*[0]
        if len(annot_offset_y) == 0: annot_offset_y = len(self.e)*[0]
        if ax == None: ax = plt.gca()
            
        for i,edge in enumerate(self.e):
            i0 = min([edge[0],edge[1]])
            i1 = max([edge[0],edge[1]])        

            x0 = self.v[i0,0]
            y0 = self.v[i0,1]

            x1 = self.v[i1,0]
            y1 = self.v[i1,1]

            if not self.w[i]:
                if width[i] == 0:
                    continue
                ax.plot([x0, x1], [y0,y1], line_style, c=colors[i], lw=width[i])
            else:
                if x1 > x0:
                    ax.plot([x0, x1-(x_max-x_min)], [y0,y1], line_style, c=colors[i], lw=width[i])
                    ax.plot([x0+(x_max-x_min), x1], [y0,y1], line_style, c=colors[i], lw=width[i])
                else:
                    ax.plot([x0-(x_max-x_min), x1], [y0,y1], line_style, c=colors[i], lw=width[i])
                    ax.plot([x0, x1+(x_max-x_min)], [y0,y1], line_style, c=colors[i], lw=width[i])                
            if len(directions):
                if directions[i] == 1:
                    # if len(self.w):
                    if self.w[i] == 0:
                        ax.arrow(x0,y0,(x1-x0)/2,(y1-y0)/2,head_width=head_width,lw=0,fc='k',ec='k',zorder=100)                    
                elif directions[i] == -1:
                    # if len(self.w):
                    if self.w[i] == 0:
                        ax.arrow(x1,y1,(x0-x1)/2,(y0-y1)/2,head_width=head_width,lw=0,fc='k',ec='k',zorder=100)

            if annotate == True:
                if publication == True:
                    label = str(i+1)
                else:
                    label = str(i)
                if self.w[i] == 0:
                    ax.annotate(label,(x0+(x1-x0)/2+annot_offset_x[i],y0+(y1-y0)/2+annot_offset_y[i]),fontsize=16,ha='center',va='center')
                else:
                    ax.annotate(label,(np.max((x0,x1)),y1),fontsize=16,color='r',ha='center',va='center')

        for node in range(len(self.v)):
            x = self.v[node,0]
            y = self.v[node,1]
            ax.plot(x,y ,'wo',mec='k',ms=ms)
            x,y = self.v[node,0], self.v[node, 1]
            if annotate:
                if publication == True:
                    label = str(node+1)
                else:
                    label = str(node)                
                ax.text(x,y,label,fontsize=12,ha='center',va='center')

        ax.set_xticks([])
        ax.set_yticks([])    

        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)        

        ax.set_aspect('equal')                

    def plot_alpha_space(self,lim=2):
        if self.bc_type != "flow":
            return
        
        if self.k == 1:
            for i in range(self.nVessels):
                if np.abs(self.K[i,0]) > 1e-6:
                    plt.plot(-self.q0[i]/self.K[i,0],0,'o',ms=10+i,label=r'$q_{'+str(i)+'} = 0$',zorder=self.nNodes-i)  

            plt.xlabel(r'$\alpha$')
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        elif self.k == 2:
            a = [np.linspace(-lim,lim,4), np.linspace(-lim,lim,4)]
            for i in self.interiorFlows:
                # if np.abs(self.K[i,1]) > 1e-6:
                plt.plot(a[0],(-a[0]*self.K[i,0]-self.q0[i])/self.K[i,1],'--',label=r'$q_{'+str(i)+'} = 0$')  
            plt.xlim((-lim,lim))
            plt.ylim((-lim,lim))
    
            plt.plot([-lim,lim],[0,0],'k',alpha=0.1,zorder=-1)
            plt.plot([0,0],[-lim,lim],'k',alpha=0.1,zorder=-1)
    
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
            plt.xlabel(r'$\alpha_1$')
            plt.ylabel(r'$\alpha_2$')
            plt.gca().set_aspect('equal')
        else:
            print('Cannot (yet) plot Kirchoff space for k > 2.')    
    
    #########################
    ###### I/O METHODS ######
    #########################    
    
    def save(self,prefix):
        scipy.io.savemat(f'data/{prefix}_eqs.mat',{'eqs':self.equilibria})
        f = open(f'data/{prefix}_params.p','wb')
        pickle.dump(self.params,f)
        f.close()