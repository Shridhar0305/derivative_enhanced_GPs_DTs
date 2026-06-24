import numpy as np
from abc import ABC, abstractmethod

from meas import PointMeasurement,PointIndexMeasurement,ΔδPointMeasurement,ΔΔΔPointMeasurement,ΔΔδPointMeasurement
from meas import dPointMeasurement,ddPointMeasurement,dddPointMeasurement,ddddPointMeasurement
import numdifftools as nd
from meas import AbstractMeasurement,AbstractPointMeasurement



# --- Helper Functions ---
def _jacobian(f, x, h=1e-8):
    """Computes the Jacobian of f at x using central differences."""
    x = np.asarray(x)  # Ensure x is an array
    fx = np.asarray(f(x))  # Ensure fx is an array
    n = len(x)
    m = fx.size # change here, use size instead of len
    J = np.zeros((m, n))
    for i in range(n):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        J[:, i] = (np.asarray(f(x_plus_h)) - fx).reshape(m) / h   # Make f(x_plus_h) an array, and reshape
    return J

def _hessian(f, x, h=1e-5):
    """Computes the Hessian of f at x using central differences."""
    x = np.asarray(x)
    fx = np.asarray(f(x))
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        H[i, :] = (_jacobian(f, x_plus_h, h) - _jacobian(f, x, h)) / h
    return (H + H.T) / 2


# Covariance Functions
class AbstractCovarianceFunction(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass

    def batched_call(self, out, x_vec, y_vec):
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = self(x_vec[i], y_vec[j])

    def batched_call_symmetric(self, out, x_vec):
        for i in range(out.shape[0]):
            for j in range(i, out.shape[1]):
                out[i, j] = self(x_vec[i], x_vec[j])
                if i != j:
                    out[j, i] = out[i, j]

# To solve PDE problems - This implementation is unverified
class MaternCovariance5_2(AbstractCovarianceFunction):
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, a: AbstractMeasurement, b: AbstractMeasurement):
        """Compute the Matern 5/2 covariance between two measurement points."""
        x = a.coordinate
        y = b.coordinate
        dist = np.linalg.norm(x-y)

        val = (1 + np.sqrt(5) * dist / self.length_scale
            + 5 * dist**2 / (3 * self.length_scale**2))
        
        exp_term =  np.exp(-np.sqrt(5) * dist / self.length_scale)

        if isinstance(a,PointMeasurement) and isinstance(b,PointMeasurement):
            return np.array(val * exp_term)

        elif isinstance(a, ΔδPointMeasurement) and isinstance(b,ΔδPointMeasurement):
            try:
                d = len(x)
            except:
                d=1
            w1_x, w2_x = a.weight_Δ, a.weight_δ
            w1_y, w2_y = b.weight_Δ, b.weight_δ

            def F(t, a):
                return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2))*np.exp(-np.sqrt(5)*t/a)
            def D2F(t, a):
                return -5*(d*a**2 + np.sqrt(5)*d*a*t - 5*t**2)/(3*a**4) * np.exp(-np.sqrt(5)*t/a)
            def D4F(t, a):
                return 25*(d*(d+2)*a**2 - (3+2*d)*np.sqrt(5)*a*t + 5*t**2)/(3*a**6) * np.exp(-np.sqrt(5)*t/a)

            return np.array(w1_x*w1_y*D4F(dist,self.length_scale) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,self.length_scale) + w2_x*w2_y*F(dist,self.length_scale))
        
        
        elif isinstance(a, ΔΔδPointMeasurement) and isinstance(b,ΔΔδPointMeasurement):
            try:
                d = len(x)
            except:
                d=1
            w1_x, w2_x, wg_x = a.weight_Δ, a.weight_δ, a.weight_ΔΔ
            w1_y, w2_y, wg_y = b.weight_Δ, b.weight_δ, b.weight_ΔΔ

            def F(t, a):
                return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2)) * np.exp(-np.sqrt(5)*t/a)
            def D2F(t, a):
                return -5*(d*a**2 + np.sqrt(5)*d*a*t - 5*t**2)/(3*a**4) * np.exp(-np.sqrt(5)*t/a)
            def D4F(t, a):
                return 25*(d*(d+2)*a**2 - (3+2*d)*np.sqrt(5)*a*t + 5*t**2)/(3*a**6) * np.exp(-np.sqrt(5)*t/a)
            def DF(t,a):
                return -5*(a+np.sqrt(5)*t)*np.exp(-np.sqrt(5)*t/a)/(3*a**3)
            def D3F(t,a):
                return 25*np.exp(-np.sqrt(5)*t/a)*(a*(2+d)-np.sqrt(5)*t)/(3*a**5)
            def DDF(t,a):
                # return 25*np.exp(-np.sqrt(5)*t/a)*(a*(d-1)-np.sqrt(5)*t)/(3*a**5)
                return 25*np.exp(-np.sqrt(5)*t/a)/(3*a**5) # corrected
            vec = a.coordinate - b.coordinate
            dist = np.linalg.norm(vec)

            return np.array(w1_x*w1_y*D4F(dist,self.length_scale) + (w2_x*w1_y+w1_x*w2_y)*D2F(dist,self.length_scale) + w2_x*w2_y*F(dist,self.length_scale) \
                 - w1_x*D3F(dist,self.length_scale)*np.sum(vec*wg_y) + w1_y*D3F(dist,self.length_scale)*np.sum(vec*wg_x) \
                 - w2_x*DF(dist,self.length_scale)*np.sum(vec*wg_y)  + w2_y*DF(dist,self.length_scale)*np.sum(vec*wg_x) \
                 + (np.sum(-wg_x*wg_y)*DF(dist,self.length_scale) + np.sum(wg_x*vec)*np.sum(wg_y*vec)*DDF(dist,self.length_scale))) # corrected
        
        elif isinstance(a, ΔΔΔPointMeasurement) and isinstance(b,ΔΔΔPointMeasurement):
            
            def F(t,a):
                return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2))*np.exp(-np.sqrt(5)*t/a)

            def F_xy(x, y):
                eps = 1e-8
                dist = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2 + eps) # Added eps here
                return F(dist, self.length_scale)
            
            def Hx_F(x,y):
                hessian = nd.Hessian(lambda x_: F_xy(x_, y))(x)
                return np.array([hessian[0, 0], hessian[0, 1], hessian[1, 1]])

            def HxHy_F(x,y):
                jacobian_matrix = nd.Jacobian(lambda y_: nd.Jacobian(lambda y__: Hx_F(x,y__))(y_))(y)
                return jacobian_matrix.flatten()[[0, 1, 2, 3, 4, 5, 9, 10, 11]]

            vec = HxHy_F(a.coordinate, b.coordinate)
            wx = np.array([a.weight_Δ11, a.weight_Δ12, a.weight_Δ22])
            wy = np.array([b.weight_Δ11, b.weight_Δ12, b.weight_Δ22])
            return  np.array(np.dot(wx, np.dot(vec.reshape((3, 3)), wy)))
        
        else: # Mixed case
            if isinstance(b,PointMeasurement):
                
                a,b = b,a # ensure a is PointMeasurement
            x = a.coordinate
            y = b.coordinate
            dist = np.linalg.norm(x-y)
            try:
                d = len(x)
            except:
                d=1
            if isinstance(b, ΔδPointMeasurement):
                
                w1_y, w2_y = b.weight_Δ, b.weight_δ
                def F(t, a):
                    return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2))*np.exp(-np.sqrt(5)*t/a)
                def D2F(t, a):
                    return -5*(d*a**2 + np.sqrt(5)*d*a*t - 5*t**2)/(3*a**4) * np.exp(-np.sqrt(5)*t/a)
                
                if dist==0:
                    return np.array(-5*d*w1_y/(3*self.length_scale**2) + w2_y)

                return np.array(w1_y*D2F(dist,self.length_scale) + w2_y * F(dist, self.length_scale))
            
            elif isinstance(b, ΔΔδPointMeasurement):
                
                w1_y, w2_y, wg_y = b.weight_Δ, b.weight_δ, b.weight_ΔΔ
                def F(t, a):
                    return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2)) * np.exp(-np.sqrt(5)*t/a)
                def D2F(t, a):
                    return -5*(d*a**2 + np.sqrt(5)*d*a*t - 5*t**2)/(3*a**4) * np.exp(-np.sqrt(5)*t/a)
                def DF(t,a):
                    return -5*(a+np.sqrt(5)*t)*np.exp(-np.sqrt(5)*t/a)/(3*a**3)
                def D3F(t,a):
                    return 25*np.exp(-np.sqrt(5)*t/a)*(a*(2+d)-np.sqrt(5)*t)/(3*a**5)
                def DDF(t,a):
                    return 25*np.exp(-np.sqrt(5)*t/a)*(a*(d-1)-np.sqrt(5)*t)/(3*a**5)
                vec = a.coordinate - b.coordinate
                dist = np.linalg.norm(vec)
                return np.array(w1_y*D3F(dist, self.length_scale)*np.sum(vec*wg_y) + w2_y*DF(dist,self.length_scale)*np.sum(vec*wg_y))
            elif isinstance(b, ΔΔΔPointMeasurement):
                
                def F(x,y,a):
                    eps = 1e-8
                    t = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+eps)
                    return (1 + np.sqrt(5)*t/a + 5*t**2/(3*a**2))*np.exp(-np.sqrt(5)*t/a)

                def Hx_F(x, y, a):
                    # Corrected to compute only the necessary elements
                    hessian = _hessian(lambda x_: F(x_, y, a), x)
                    return np.array([hessian[0, 0], hessian[0, 1], hessian[1, 1]])
                vec = Hx_F(b.coordinate, a.coordinate, self.length_scale)
                wy = np.array([b.weight_Δ11,b.weight_Δ12,b.weight_Δ22])
                return np.array(np.dot(vec,wy))
            else:
                raise TypeError("Unsupported measurement type for Matern covariance.")
            
# Effort to implement derivatives with Matern. (not completed)
class MaternCovariance5_2_non_pde(AbstractCovarianceFunction):
    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, a: AbstractMeasurement, b: AbstractMeasurement):
        """Compute the Matern 5/2 covariance between two measurement points."""
        x = a.coordinate
        y = b.coordinate
        dist = np.linalg.norm(x-y)

        val = (1 + np.sqrt(5) * dist / self.length_scale
            + 5 * dist**2 / (3 * self.length_scale**2))
        
        exp_term =  np.exp(-np.sqrt(5) * dist / self.length_scale)

        if isinstance(a,PointMeasurement) and isinstance(b,PointMeasurement):
            return np.array(val * exp_term)
        elif isinstance(a, dPointMeasurement) and isinstance(b,dPointMeasurement):
            d=len(x)

            
            def D2F(t, a_ls):
                return -5*(d*a_ls**2 + np.sqrt(5)*d*a_ls*t - 5*t**2)/(3*a_ls**4) * np.exp(-np.sqrt(5)*t/a_ls)


            return D2F(dist,self.length_scale)
    
        elif isinstance(a, ddPointMeasurement) and isinstance(b,ddPointMeasurement):
            d=len(x)
            def D4F(t, a):
                return 25*(d*(d+2)*a**2 - (3+2*d)*np.sqrt(5)*a*t + 5*t**2)/(3*a**6) * np.exp(-np.sqrt(5)*t/a)

            return D4F(dist,self.length_scale)
        
        elif (isinstance(a, PointMeasurement) and isinstance(b,dPointMeasurement)) or (isinstance(a, dPointMeasurement) and isinstance(b,PointMeasurement)):

            
            def DF(t,a_ls):
                return -5*t*(a_ls+np.sqrt(5)*t)*np.exp(-np.sqrt(5)*t/a_ls)/(3*a_ls**3)

            return DF(dist,self.length_scale)
        
        elif isinstance(a, PointMeasurement) and isinstance(b,ddPointMeasurement):

            def D2F(t, a):
                return -5*(d*a**2 + np.sqrt(5)*d*a*t - 5*t**2)/(3*a**4) * np.exp(-np.sqrt(5)*t/a)

            return D2F(dist,self.length_scale)
        
        elif isinstance(a, dPointMeasurement) and isinstance(b,ddPointMeasurement):
            def D3F(t,a):
                return 25*np.exp(-np.sqrt(5)*t/a)*(a*(2+d)-np.sqrt(5)*t)/(3*a**5)

            return D3F(dist,self.length_scale)
    
        else:
                raise TypeError("Unsupported measurement type for Matern covariance non PDE.")


# Gaussian kernel with upto 4th order implemented (tested up to 4th order on Multi-dimensional Scalable function)
class GaussianCovariance_generic():

    def __init__(self, length_scale):
        self.length_scale = length_scale


    def __call__(self, a_inp:AbstractMeasurement, b_inp:AbstractMeasurement):
        """Compute the Gaussian covariance between two measurement points."""
        x = a_inp.coordinate
        y = b_inp.coordinate
        dist_sq = np.sum((x - y)**2)
        l2 = self.length_scale**2
        l4 = self.length_scale**4
        l6 = self.length_scale**6
        l8 = self.length_scale**8
        l10 = self.length_scale**10
        l12 = self.length_scale**12
        exp_term = np.exp(-0.5 * dist_sq / l2)
        diff = x - y
        d_dim = len(x)

        if isinstance(a_inp, PointMeasurement) and isinstance(b_inp, PointMeasurement):
            return exp_term

        elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, dPointMeasurement):
            index_a = a_inp.derivative_index
            index_b = b_inp.derivative_index

            if index_a == index_b:
                return (1/l2 - diff[index_a] * diff[index_b] / l4) * exp_term
            else:
                return (- diff[index_a] * diff[index_b] / l4) * exp_term


        elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):

            index_a1, index_a2 = a_inp.derivative_indices
            index_b1, index_b2 = b_inp.derivative_indices

            delta_ik = 1 if index_a1 == index_b1 else 0
            delta_il = 1 if index_a1 == index_b2 else 0
            delta_jk = 1 if index_a2 == index_b1 else 0
            delta_jl = 1 if index_a2 == index_b2 else 0
            delta_ij = 1 if index_a1 == index_a2 else 0
            delta_kl = 1 if index_b1 == index_b2 else 0

            term_val = (delta_ik * delta_jl + delta_il * delta_jk + delta_ij * delta_kl) / l4
            term_val -= (delta_ik * diff[index_a2] * diff[index_b2] +
                        delta_il * diff[index_a2] * diff[index_b1] +
                        delta_ij * diff[index_b1] * diff[index_b2] +
                        delta_jk * diff[index_a1] * diff[index_b2] +
                        delta_jl * diff[index_a1] * diff[index_b1] +
                        delta_kl * diff[index_a1] * diff[index_a2]) / l6
            term_val += (diff[index_a1] * diff[index_a2] * diff[index_b1] * diff[index_b2]) / l8

            return term_val * exp_term

        elif (isinstance(a_inp, PointMeasurement) and isinstance(b_inp, dPointMeasurement)):
            index_b = b_inp.derivative_index
            return ((diff[index_b]) / l2) * exp_term

        elif (isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, PointMeasurement)):
            index_a = a_inp.derivative_index
            return (-(diff[index_a]) / l2) * exp_term

        elif isinstance(a_inp, PointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            index_b1, index_b2 = b_inp.derivative_indices
            d_ij=1 if index_b1==index_b2 else 0
            term = (-d_ij/l2 + (diff[index_b1]*diff[index_b2]) / l4)

            return term * exp_term

        elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, PointMeasurement):
            index_a1, index_a2 = a_inp.derivative_indices
            d_ij=1 if index_a1==index_a2 else 0
            term = (-d_ij/l2 + (diff[index_a1]*diff[index_a2]) / l4)
            return term * exp_term

        elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            index_a = a_inp.derivative_index
            index_b1, index_b2 = b_inp.derivative_indices

            delta_ik = 1 if index_a == index_b2 else 0
            delta_jk = 1 if index_b1 == index_b2 else 0
            delta_ij = 1 if index_a == index_b1 else 0

            term_val =  (
                (delta_ik * diff[index_b1] +
                delta_jk * diff[index_a] +
                delta_ij * diff[index_b2]) / l4 -
                (diff[index_a] * diff[index_b1] * diff[index_b2]) / l6
            )


            return term_val * exp_term

        elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
            index_b = b_inp.derivative_index
            index_a1, index_a2 = a_inp.derivative_indices


            delta_ik = 1 if index_a1 == index_b else 0
            delta_jk = 1 if index_a2 == index_b else 0
            delta_ij = 1 if index_a1 == index_a2 else 0

            term_val = (
                -(delta_ik * diff[index_a2] +
                delta_jk * diff[index_a1] +
                delta_ij * diff[index_b]) / l4 +
                (diff[index_a1] * diff[index_a2] * diff[index_b]) / l6
            )

            return term_val * exp_term

       
        # Handling dddPointMeasurement
        elif isinstance(a_inp, PointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            index_b1, index_b2, index_b3 = b_inp.derivative_indices
            delta_ik = 1 if index_b1 == index_b2 else 0
            delta_jk = 1 if index_b1 == index_b3 else 0
            delta_ij = 1 if index_b2 == index_b3 else 0
            term = (diff[index_b1] * diff[index_b2] * diff[index_b3] / l6)
            term+=(-(delta_ik * diff[index_b3] +
                delta_jk * diff[index_b2] +
                delta_ij * diff[index_b1]) / l4)

            return term * exp_term

        elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, PointMeasurement):
            index_a1, index_a2, index_a3 = a_inp.derivative_indices
            delta_ik = 1 if index_a1 == index_a2 else 0
            delta_jk = 1 if index_a1 == index_a3 else 0
            delta_ij = 1 if index_a2 == index_a3 else 0

            term = (-diff[index_a1] * diff[index_a2] * diff[index_a3] / l6)
            term+=((delta_ik * diff[index_a3] +
                delta_jk * diff[index_a2] +
                delta_ij * diff[index_a1]) / l4)

            return term * exp_term

        elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            a = a_inp.derivative_index
            b1, b2, b3 = b_inp.derivative_indices
            d_ik = 1 if b1 == b2 else 0
            d_jk = 1 if b1 == b3 else 0
            d_ij = 1 if b2 == b3 else 0
            d_kl = 1 if a == b1 else 0
            d_il = 1 if a == b2 else 0
            d_jl = 1 if a ==b3 else 0
            t3 = (diff[a] * diff[b1] * diff[b2] * diff[b3])
            t1=(d_kl*d_ij+d_il*d_jk+d_jl*d_ik)
            t2=((d_ij*diff[b1]*diff[a])+(d_jk*diff[b2]*diff[a])+(d_ik*diff[b3]*diff[a])+
                (d_il*diff[b3]*diff[b1])+(d_jl*diff[b2]*diff[b1])+(d_kl*diff[b2]*diff[b3]))
            
            term=(-t1/l4+t2/l6-t3/l8)
            return term * exp_term

        elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
            a = b_inp.derivative_index
            b1, b2, b3 = a_inp.derivative_indices
            d_ik = 1 if b1 == b2 else 0
            d_jk = 1 if b1 == b3 else 0
            d_ij = 1 if b2 == b3 else 0
            d_kl = 1 if a == b1 else 0
            d_il = 1 if a == b2 else 0
            d_jl = 1 if a ==b3 else 0
            t3 = (diff[a] * diff[b1] * diff[b2] * diff[b3])
            t1=(d_kl*d_ij+d_il*d_jk+d_jl*d_ik)
            t2=((d_ij*diff[b1]*diff[a])+(d_jk*diff[b2]*diff[a])+(d_ik*diff[b3]*diff[a])+
                (d_il*diff[b3]*diff[b1])+(d_jl*diff[b2]*diff[b1])+(d_kl*diff[b2]*diff[b3]))
            term=(-t1/l4+t2/l6-t3/l8)

            return term * exp_term

        elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            a1, a2 = a_inp.derivative_indices
            b1, b2, b3 = b_inp.derivative_indices
            d_ij=1 if b2==b3 else 0
            d_il=1 if b2==a2 else 0
            d_ip=1 if b2==a1 else 0
            d_ik=1 if b2==b1 else 0
            d_jl=1 if b3==a2 else 0
            d_jk=1 if b3==b1 else 0
            d_jp=1 if b3==a1 else 0
            d_kl=1 if b1==a2 else 0
            d_kp=1 if b1==a1 else 0
            d_lp=1 if a1==a2 else 0

            t3 = diff[a1] * diff[a2] * diff[b1] * diff[b2] * diff[b3]
            t2 = ((d_ip*diff[b3]*diff[b1]*diff[a2])+(d_jp*diff[b2]*diff[b1]*diff[a2])+
                  (d_kp*diff[b2]*diff[b3]*diff[a2])+(d_lp*diff[b2]*diff[b3]*diff[b1])+
                  (d_ij*diff[b1]*diff[a2]*diff[a1])+(d_ik*diff[b3]*diff[a2]*diff[a1])+
                  (d_jk*diff[b2]*diff[a2]*diff[a1])+(d_il*diff[b3]*diff[b1]*diff[a1])+
                  (d_jl*diff[b2]*diff[b1]*diff[a1])+(d_kl*diff[b2]*diff[b3]*diff[a1]))
            
            t1=((d_ij*d_kl+d_jl*d_ik+d_il*d_jk)*diff[a1]+(d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*diff[a2]+
                (d_ij*d_lp+d_il*d_jp+d_jl*d_ip)*diff[b1]+(d_jk*d_lp+d_jl*d_kp+d_kl*d_jp)*diff[b2]+
                (d_ik*d_lp+d_il*d_kp+d_kl*d_ip)*diff[b3])
            term=t1/l6-t2/l8+t3/l10
            
            return term * exp_term

        elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            a1, a2 = b_inp.derivative_indices
            b1, b2, b3 = a_inp.derivative_indices
            d_ij=1 if b2==b3 else 0
            d_il=1 if b2==a2 else 0
            d_ip=1 if b2==a1 else 0
            d_ik=1 if b2==b1 else 0
            d_jl=1 if b3==a2 else 0
            d_jk=1 if b3==b1 else 0
            d_jp=1 if b3==a1 else 0
            d_kl=1 if b1==a2 else 0
            d_kp=1 if b1==a1 else 0
            d_lp=1 if a1==a2 else 0

            t3 = diff[a1] * diff[a2] * diff[b1] * diff[b2] * diff[b3]
            t2 = ((d_ip*diff[b3]*diff[b1]*diff[a2])+(d_jp*diff[b2]*diff[b1]*diff[a2])+
                  (d_kp*diff[b2]*diff[b3]*diff[a2])+(d_lp*diff[b2]*diff[b3]*diff[b1])+
                  (d_ij*diff[b1]*diff[a2]*diff[a1])+(d_ik*diff[b3]*diff[a2]*diff[a1])+
                  (d_jk*diff[b2]*diff[a2]*diff[a1])+(d_il*diff[b3]*diff[b1]*diff[a1])+
                  (d_jl*diff[b2]*diff[b1]*diff[a1])+(d_kl*diff[b2]*diff[b3]*diff[a1]))
            t1=((d_ij*d_kl+d_jl*d_ik+d_il*d_jk)*diff[a1]+(d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*diff[a2]+
                (d_ij*d_lp+d_il*d_jp+d_jl*d_ip)*diff[b1]+(d_jk*d_lp+d_jl*d_kp+d_kl*d_jp)*diff[b2]+
                (d_ik*d_lp+d_il*d_kp+d_kl*d_ip)*diff[b3])
            term=-t1/l6+t2/l8-t3/l10
            
            return term * exp_term

        
        elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):

            a1, a2, a3 = a_inp.derivative_indices
            b1, b2, b3 = b_inp.derivative_indices

            d_ij=1 if b2==b3 else 0
            d_ik=1 if b2==b1 else 0
            d_iq=1 if b2==a1 else 0
            d_ip=1 if b2==a2 else 0
            d_il=1 if b2==a3 else 0
            d_jk=1 if b3==b1 else 0
            d_jq=1 if b3==a1 else 0
            d_jp=1 if b3==a2 else 0
            d_jl=1 if b3==a3 else 0
            d_kq=1 if b1==a1 else 0
            d_kp=1 if b1==a2 else 0
            d_kl=1 if b1==a3 else 0
            d_pq=1 if a1==a2 else 0
            d_lq=1 if a1==a3 else 0
            d_lp=1 if a2==a3 else 0
            
            t1=((d_ij*d_kl+d_jl*d_ik+d_il*d_jk)*d_pq+(d_jk*d_lp+d_jl*d_kp+d_kl*d_jp)*d_iq+
                (d_ik*d_lp+d_il*d_kp+d_kl*d_ip)*d_jq+(d_ij*d_lp+d_il*d_jp+d_jl*d_ip)*d_kq+
                (d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*d_lq)
            
            t2=((d_ij*d_kl+d_jl*d_ik+d_il*d_jk)*diff[a2]*diff[a1]+(d_jk*d_lp+d_jl*d_kp+d_kl*d_jp)*diff[b2]*diff[a1]+
                (d_ik*d_lp+d_il*d_kp+d_kl*d_ip)*diff[b3]*diff[a1]+(d_ij*d_lp+d_il*d_jp+d_jl*d_ip)*diff[a1]*diff[b1]+
                (d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*diff[a3]*diff[a1]+(d_ip*d_jq+d_jp*d_iq+d_ij*d_pq)*diff[b1]*diff[a3]+
                (d_ip*d_kq+d_kp*d_iq+d_ik*d_pq)*diff[a3]*diff[b3]+(d_ip*d_lq+d_lp*d_iq+d_il*d_pq)*diff[b3]*diff[b1]+
                (d_jp*d_kq+d_kp*d_jq+d_jk*d_pq)*diff[b2]*diff[a3]+(d_jp*d_lq+d_lp*d_jq+d_jl*d_pq)*diff[b2]*diff[b1]+
                (d_kp*d_lq+d_lp*d_kq+d_kl*d_pq)*diff[b2]*diff[b3]+(d_ij*d_kq+d_jk*d_iq+d_ik*d_jq)*diff[a3]*diff[a2]+
                (d_ij*d_lq+d_jl*d_iq+d_il*d_jq)*diff[b1]*diff[a2]+(d_ik*d_lq+d_il*d_kq+d_kl*d_iq)*diff[b3]*diff[a2]+
                (d_jk*d_lq+d_jl*d_kq+d_kl*d_jq)*diff[b2]*diff[a2]
                )
            
            t3=((d_ip*diff[b3]*diff[b1]*diff[a3]*diff[a1])+(d_jp*diff[b2]*diff[b1]*diff[a3]*diff[a1])+
                  (d_kp*diff[b2]*diff[b3]*diff[a3]*diff[a1])+(d_lp*diff[b2]*diff[b3]*diff[b1]*diff[a1])+
                  (d_ij*diff[b1]*diff[a3]*diff[a2]*diff[a1])+(d_ik*diff[b3]*diff[a3]*diff[a2]*diff[a1])+
                  (d_jk*diff[b2]*diff[a3]*diff[a2]*diff[a1])+(d_il*diff[b3]*diff[b1]*diff[a2]*diff[a1])+
                  (d_jl*diff[b2]*diff[b1]*diff[a2]*diff[a1])+(d_kl*diff[b2]*diff[b3]*diff[a2]*diff[a1])+
                  (d_iq*diff[b3]*diff[b1]*diff[a3]*diff[a2])+(d_jq*diff[a2]*diff[a3]*diff[b1]*diff[b2])+
                  (d_kq*diff[b2]*diff[b3]*diff[a2]*diff[a3])+(d_lq*diff[a2]*diff[b3]*diff[b1]*diff[b2])+
                  (d_pq*diff[a3]*diff[b3]*diff[b1]*diff[b2]))
            t4=diff[a1]*diff[a2]*diff[a3]*diff[b1]*diff[b2]*diff[b3]
            term=t1/l6-t2/l8+t3/l10-t4/l12

            return term * exp_term



        # # Handling ddddPointMeasurement - Placeholder as the formulas are very complex
        # defining the calculations as function so that I can reuse in subsequent calculations
        def pd4(ind,dif):
            b1, b2, b3, b4 = ind
            d_ij=1 if b1==b2 else 0
            d_kl=1 if b3==b4 else 0
            d_ik=1 if b1==b3 else 0
            d_jl=1 if b2==b4 else 0
            d_jk=1 if b2==b3 else 0
            d_il=1 if b1==b4 else 0

            term = (d_ij*d_kl+d_ik*d_jl+d_jk*d_il)/l4-((d_ik*dif[b2]+d_jk*dif[b1]+d_ij*dif[b3])*dif[b4]/l6)-(
                    (d_il*dif[b2]+d_jl*dif[b1])*dif[b3]/l6)-(d_kl*dif[b1]*dif[b2])/l6+(dif[b1]*dif[b2]*dif[b3]*dif[b4])/l8
            return term
        
        def dd4(ind1,ind2,dif):
            a1 =ind1
            b1, b2, b3, b4=ind2
            d_ij=1 if b1==b2 else 0
            d_kl=1 if b3==b4 else 0
            d_ik=1 if b1==b3 else 0
            d_jl=1 if b2==b4 else 0
            d_jk=1 if b2==b3 else 0
            d_il=1 if b1==b4 else 0
            d_im=1 if a1==b1 else 0
            d_jm=1 if a1==b2 else 0
            d_km=1 if a1==b3 else 0
            d_lm=1 if a1==b4 else 0

            term1=-(dif[a1]/l2)*pd4(ind2,dif=dif)
            term2=-((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*dif[b1]+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*dif[b2]+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*dif[b3]
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*dif[b4])/l6
            term3=(d_im*dif[b2]*dif[b3]*dif[b4]+d_jm*dif[b1]*dif[b3]*dif[b4]
                    +d_km*dif[b1]*dif[b2]*dif[b4]+d_lm*dif[b1]*dif[b2]*dif[b3])/l8
            term = term1 +term2+term3
            return term
        
        def d2d4(ind1,ind2,dif):
            a1,a2=ind1
            # print("a1,a2",a1,a2)
            b1, b2, b3, b4=ind2
            d_ij=1 if b1==b2 else 0
            d_kl=1 if b3==b4 else 0
            d_ik=1 if b1==b3 else 0
            d_jl=1 if b2==b4 else 0
            d_jk=1 if b2==b3 else 0
            d_il=1 if b1==b4 else 0
            d_im=1 if a1==b1 else 0
            d_jm=1 if a1==b2 else 0
            d_km=1 if a1==b3 else 0
            d_lm=1 if a1==b4 else 0
            d_in=1 if a2==b1 else 0
            d_jn=1 if a2==b2 else 0
            d_kn=1 if a2==b3 else 0
            d_ln=1 if a2==b4 else 0
            d_mn=1 if a1==a2 else 0

            term1=(-dif[a2]*dd4(a1,ind2,dif)/l2)-pd4(ind2,dif)*d_mn/l2
            term2=(-dif[a1]/l2)*(
                -((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*dif[b1]+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*dif[b2]+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*dif[b3]
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*dif[b4])/l6+
                    ((d_in*dif[b2]*dif[b3]*dif[b4])+(d_jn*dif[b1]*dif[b3]*dif[b4])+(d_kn*dif[b1]*dif[b2]*dif[b4])
                     +(d_ln*dif[b1]*dif[b2]*dif[b3]))/l8)
            term3=-((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*d_in+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*d_jn+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*d_kn
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*d_ln)/l6
            term4=(d_im*(d_jn*dif[b3]*dif[b4]+d_kn*dif[b2]*dif[b4]+d_ln*dif[b2]*dif[b3])+
                   d_jm*(d_in*dif[b3]*dif[b4]+d_kn*dif[b1]*dif[b4]+d_ln*dif[b1]*dif[b3])+
                   d_km*(d_in*dif[b2]*dif[b4]+d_jn*dif[b1]*dif[b4]+d_ln*dif[b1]*dif[b2])+
                   d_lm*(d_in*dif[b2]*dif[b3]+d_jn*dif[b1]*dif[b3]+d_kn*dif[b1]*dif[b2])
                   )/l8

            term = term1+term2+term3+term4
            return term

        def d3d4(ind1,ind2,dif):
            a1,a2,a3=ind1
            b1, b2, b3, b4=ind2
            d_ij=1 if b1==b2 else 0
            d_kl=1 if b3==b4 else 0
            d_ik=1 if b1==b3 else 0
            d_jl=1 if b2==b4 else 0
            d_jk=1 if b2==b3 else 0
            d_il=1 if b1==b4 else 0
            d_im=1 if a1==b1 else 0
            d_jm=1 if a1==b2 else 0
            d_km=1 if a1==b3 else 0
            d_lm=1 if a1==b4 else 0
            d_in=1 if a2==b1 else 0
            d_jn=1 if a2==b2 else 0
            d_kn=1 if a2==b3 else 0
            d_ln=1 if a2==b4 else 0
            d_mn=1 if a1==a2 else 0
            d_io=1 if a3==b1 else 0
            d_jo=1 if a3==b2 else 0
            d_ko=1 if a3==b3 else 0
            d_lo=1 if a3==b4 else 0
            d_mo=1 if a3==a1 else 0
            d_no=1 if a3==a2 else 0

            term1=(-dif[a3]*d2d4(ind1=[a1,a2],ind2=ind2,dif=dif)/l2)-(d_no*dd4(a1,ind2,dif)/l2)
            term2=(-dif[a2]/l2)*((-pd4(ind2,dif)*d_mo/l2)+(-dif[a1]/l2)*(-1*((d_jl*d_ko+d_kl*d_jo+d_jk*d_lo)*dif[b1]+(d_il*d_ko+d_kl*d_io+d_ik*d_lo)*dif[b2]
                    +(d_il*d_jo+d_jl*d_io+d_ij*d_lo)*dif[b3]+(d_ij*d_ko+d_ik*d_jo+d_jk*d_io)*dif[b4])/l6+
                    ((d_io*dif[b2]*dif[b3]*dif[b4])+(d_jo*dif[b1]*dif[b3]*dif[b4])+(d_ko*dif[b1]*dif[b2]*dif[b4])+
                     (d_lo*dif[b1]*dif[b2]*dif[b3]))/l8)-
                    ((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*d_io+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*d_jo+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*d_ko
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*d_lo)/l6+
                    (d_im*(d_jo*dif[b3]*dif[b4]+d_ko*dif[b2]*dif[b4]+d_lo*dif[b2]*dif[b3])+
                   d_jm*(d_io*dif[b3]*dif[b4]+d_ko*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b3])+
                   d_km*(d_io*dif[b2]*dif[b4]+d_jo*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b2])+
                   d_lm*(d_io*dif[b2]*dif[b3]+d_jo*dif[b1]*dif[b3]+d_ko*dif[b1]*dif[b2])
                   )/l8)

            term3=(-d_mn/l2)*(-1*((d_jl*d_ko+d_kl*d_jo+d_jk*d_lo)*dif[b1]+(d_il*d_ko+d_kl*d_io+d_ik*d_lo)*dif[b2]
                    +(d_il*d_jo+d_jl*d_io+d_ij*d_lo)*dif[b3]+(d_ij*d_ko+d_ik*d_jo+d_jk*d_io)*dif[b4])/l6+
                    ((d_io*dif[b2]*dif[b3]*dif[b4])+(d_jo*dif[b1]*dif[b3]*dif[b4])+(d_ko*dif[b1]*dif[b2]*dif[b4])+
                     (d_lo*dif[b1]*dif[b2]*dif[b3]))/l8)
            term4=(-d_mo/l2)*(-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*dif[b1]+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*dif[b2]+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*dif[b3]
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*dif[b4])/l6+
                    ((d_in*dif[b2]*dif[b3]*dif[b4])+(d_jn*dif[b1]*dif[b3]*dif[b4])+(d_kn*dif[b1]*dif[b2]*dif[b4])+
                     (d_ln*dif[b1]*dif[b2]*dif[b3]))/l8)
            term5=(-dif[a1]/l2)*(-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*d_io+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*d_jo+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*d_ko
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*d_lo)/l6+
                    (d_in*(d_jo*dif[b3]*dif[b4]+d_ko*dif[b2]*dif[b4]+d_lo*dif[b2]*dif[b3])+
                   d_jn*(d_io*dif[b3]*dif[b4]+d_ko*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b3])+
                   d_kn*(d_io*dif[b2]*dif[b4]+d_jo*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b2])+
                   d_ln*(d_io*dif[b2]*dif[b3]+d_jo*dif[b1]*dif[b3]+d_ko*dif[b1]*dif[b2]))/l8)
            term6=(d_im*(d_jn*(d_ko*dif[b4]+dif[b3]*d_lo)+d_kn*(d_jo*dif[b4]+dif[b2]*d_lo)+d_ln*(d_jo*dif[b3]+dif[b2]*d_ko))+
                   d_jm*(d_in*(d_ko*dif[b4]+dif[b3]*d_lo)+d_kn*(d_io*dif[b4]+dif[b1]*d_lo)+d_ln*(d_io*dif[b3]+dif[b1]*d_ko))+
                   d_km*(d_in*(d_jo*dif[b4]+dif[b2]*d_lo)+d_jn*(d_io*dif[b4]+dif[b1]*d_lo)+d_ln*(d_io*dif[b2]+dif[b1]*d_jo))+
                   d_lm*(d_in*(d_jo*dif[b3]+dif[b2]*d_ko)+d_jn*(d_io*dif[b3]+dif[b1]*d_ko)+d_kn*(d_io*dif[b2]+dif[b1]*d_jo))
                   )/l8
            term=term1+term2+term3+term4+term5+term6
            return term
        
        def d4d4(ind1,ind2,dif):
            a1,a2,a3,a4=ind1
            b1, b2, b3, b4=ind2
            d_ij=1 if b1==b2 else 0
            d_kl=1 if b3==b4 else 0
            d_ik=1 if b1==b3 else 0
            d_jl=1 if b2==b4 else 0
            d_jk=1 if b2==b3 else 0
            d_il=1 if b1==b4 else 0
            d_im=1 if a1==b1 else 0
            d_jm=1 if a1==b2 else 0
            d_km=1 if a1==b3 else 0
            d_lm=1 if a1==b4 else 0
            d_in=1 if a2==b1 else 0
            d_jn=1 if a2==b2 else 0
            d_kn=1 if a2==b3 else 0
            d_ln=1 if a2==b4 else 0
            d_mn=1 if a1==a2 else 0
            d_io=1 if a3==b1 else 0
            d_jo=1 if a3==b2 else 0
            d_ko=1 if a3==b3 else 0
            d_lo=1 if a3==b4 else 0
            d_mo=1 if a3==a1 else 0
            d_no=1 if a3==a2 else 0
            d_ip=1 if a4==b1 else 0
            d_jp=1 if a4==b2 else 0
            d_kp=1 if a4==b3 else 0
            d_lp=1 if a4==b4 else 0
            d_mp=1 if a4==a1 else 0
            d_np=1 if a4==a2 else 0
            d_po=1 if a3==a4 else 0

            pd4_xp=(-1*((d_jl*d_kp+d_kl*d_jp+d_jk*d_lp)*dif[b1]+(d_il*d_kp+d_kl*d_ip+d_ik*d_lp)*dif[b2]
                    +(d_il*d_jp+d_jl*d_ip+d_ij*d_lp)*dif[b3]+(d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*dif[b4])/l6+
                    ((d_ip*dif[b2]*dif[b3]*dif[b4])+(d_jp*dif[b1]*dif[b3]*dif[b4])+(d_kp*dif[b1]*dif[b2]*dif[b4])+
                     (d_lp*dif[b1]*dif[b2]*dif[b3]))/l8)
            
            pd4_xo=(-1*((d_jl*d_ko+d_kl*d_jo+d_jk*d_lo)*dif[b1]+(d_il*d_ko+d_kl*d_io+d_ik*d_lo)*dif[b2]
                    +(d_il*d_jo+d_jl*d_io+d_ij*d_lo)*dif[b3]+(d_ij*d_ko+d_ik*d_jo+d_jk*d_io)*dif[b4])/l6+
                    ((d_io*dif[b2]*dif[b3]*dif[b4])+(d_jo*dif[b1]*dif[b3]*dif[b4])+(d_ko*dif[b1]*dif[b2]*dif[b4])+
                     (d_lo*dif[b1]*dif[b2]*dif[b3]))/l8)
            
            dd4_xp=((-pd4(ind2,dif)*d_mp/l2)+(-dif[a1]/l2)*(pd4_xp)-
                    ((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*d_ip+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*d_jp+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*d_kp
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*d_lp)/l6+
                    (d_im*(d_jp*dif[b3]*dif[b4]+d_kp*dif[b2]*dif[b4]+d_lp*dif[b2]*dif[b3])+
                   d_jm*(d_ip*dif[b3]*dif[b4]+d_kp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b3])+
                   d_km*(d_ip*dif[b2]*dif[b4]+d_jp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b2])+
                   d_lm*(d_ip*dif[b2]*dif[b3]+d_jp*dif[b1]*dif[b3]+d_kp*dif[b1]*dif[b2])
                   )/l8)
            dd4_xo=((-pd4(ind2,dif)*d_mo/l2)+(-dif[a1]/l2)*(pd4_xo)-
                    ((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*d_io+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*d_jo+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*d_ko
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*d_lo)/l6+
                    (d_im*(d_jo*dif[b3]*dif[b4]+d_ko*dif[b2]*dif[b4]+d_lo*dif[b2]*dif[b3])+
                   d_jm*(d_io*dif[b3]*dif[b4]+d_ko*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b3])+
                   d_km*(d_io*dif[b2]*dif[b4]+d_jo*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b2])+
                   d_lm*(d_io*dif[b2]*dif[b3]+d_jo*dif[b1]*dif[b3]+d_ko*dif[b1]*dif[b2])
                   )/l8)
            pd4_xo_xp=(-1*(d_ip*(d_il*d_ko+d_kl*d_jo+d_jk*d_lo)+d_jp*(d_il*d_ko+d_kl*d_io+d_ik*d_lo)+
                        d_kp*(d_il*d_jo+d_jl*d_io+d_ij*d_lo)+d_lp*(d_ij*d_ko+d_ik*d_jo+d_jk*d_io))/l6+(
                            d_io*(d_jp*dif[b3]*dif[b4]+d_kp*dif[b2]*dif[b4]+d_lp*dif[b2]*dif[b3])+
                            d_jo*(d_ip*dif[b3]*dif[b4]+d_kp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b3])+
                            d_ko*(d_ip*dif[b2]*dif[b4]+d_jp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b2])+
                            d_lo*(d_ip*dif[b2]*dif[b3]+d_jp*dif[b1]*dif[b3]+d_kp*dif[b1]*dif[b2]))/l8)
            

            dd4_xo_xp=(((-d_mo/l2)*pd4_xp)-((d_mp/l2)*pd4_xo)-((dif[a1]/l2)*pd4_xo_xp)+
                       (d_im*(d_jo*(d_kp*dif[b4]+dif[b3]*d_lp)+d_ko*(d_jp*dif[b4]+dif[b2]*d_lp)+d_lo*(d_jp*dif[b3]+dif[b2]*d_kp))+
                        d_jm*(d_io*(d_kp*dif[b4]+dif[b3]*d_lp)+d_ko*(d_ip*dif[b4]+dif[b1]*d_lp)+d_lo*(d_ip*dif[b3]+dif[b1]*d_kp))+
                        d_km*(d_io*(d_jp*dif[b4]+dif[b2]*d_lp)+d_jo*(d_ip*dif[b4]+dif[b1]*d_lp)+d_lo*(d_ip*dif[b2]+dif[b1]*d_jp))+
                        d_lm*(d_io*(d_jp*dif[b3]+dif[b2]*d_kp)+d_jo*(d_ip*dif[b3]+dif[b1]*d_kp)+d_ko*(d_ip*dif[b2]+dif[b1]*d_jp)))/l8)


            d2d4_xp=(-dd4(a1,ind2,dif)*(d_np/l2))-((dif[a2]/l2)*dd4_xp)-((d_mn/l2)*pd4_xp)
            d2d4_xp+=((-d_mp/l2)*(-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*dif[b1]+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*dif[b2]+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*dif[b3]
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*dif[b4])/l6+
                    ((d_in*dif[b2]*dif[b3]*dif[b4])+(d_jn*dif[b1]*dif[b3]*dif[b4])+(d_kn*dif[b1]*dif[b2]*dif[b4])+
                     (d_ln*dif[b1]*dif[b2]*dif[b3]))/l8))
            d2d4_xp+=((-dif[a1]/l2)*(
                    (-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*d_ip+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*d_jp+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*d_kp
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*d_lp)/l6+
                    (d_in*(d_jp*dif[b3]*dif[b4]+d_kp*dif[b2]*dif[b4]+d_lp*dif[b2]*dif[b3])+
                   d_jn*(d_ip*dif[b3]*dif[b4]+d_kp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b3])+
                   d_kn*(d_ip*dif[b2]*dif[b4]+d_jp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b2])+
                   d_ln*(d_ip*dif[b2]*dif[b3]+d_jp*dif[b1]*dif[b3]+d_kp*dif[b1]*dif[b2]))/l8)))
            d2d4_xp+=((d_im*(d_jn*(d_kp*dif[b4]+dif[b3]*d_lp)+d_kn*(d_jp*dif[b4]+dif[b2]*d_lp)+d_ln*(d_jp*dif[b3]+dif[b2]*d_kp))+
                   d_jm*(d_in*(d_kp*dif[b4]+dif[b3]*d_lp)+d_kn*(d_ip*dif[b4]+dif[b1]*d_lp)+d_ln*(d_ip*dif[b3]+dif[b1]*d_kp))+
                   d_km*(d_in*(d_jp*dif[b4]+dif[b2]*d_lp)+d_jn*(d_ip*dif[b4]+dif[b1]*d_lp)+d_ln*(d_ip*dif[b2]+dif[b1]*d_jp))+
                   d_lm*(d_in*(d_jp*dif[b3]+dif[b2]*d_kp)+d_jn*(d_ip*dif[b3]+dif[b1]*d_kp)+d_kn*(d_ip*dif[b2]+dif[b1]*d_jp))
                   )/l8)
            
            term1=((-d3d4(ind1=[a1,a2,a3],ind2=ind2,dif=dif)*(dif[a4]/l2))-((d_po/l2)*d2d4(ind1=[a1,a2],ind2=ind2,dif=dif))-
                   ((dif[a3]/l2)*d2d4_xp)-((d_no/l2)*(dd4_xp))-((d_np/l2)*dd4_xo) - ((dif[a2]/l2)*dd4_xo_xp)-((d_mn/l2)*pd4_xo_xp))

            
            term2= ((-d_mo/l2)*(-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*d_ip+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*d_jp+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*d_kp
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*d_lp)/l6+
                    (d_in*(d_jp*dif[b3]*dif[b4]+d_kp*dif[b2]*dif[b4]+d_lp*dif[b2]*dif[b3])+
                   d_jn*(d_ip*dif[b3]*dif[b4]+d_kp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b3])+
                   d_kn*(d_ip*dif[b2]*dif[b4]+d_jp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b2])+
                   d_ln*(d_ip*dif[b2]*dif[b3]+d_jp*dif[b1]*dif[b3]+d_kp*dif[b1]*dif[b2]))/l8))

            
            term3=((-d_mp/l2)*(-1*((d_jl*d_kn+d_kl*d_jn+d_jk*d_ln)*d_io+(d_il*d_kn+d_kl*d_in+d_ik*d_ln)*d_jo+(d_il*d_jn+d_jl*d_in+d_ij*d_ln)*d_ko
                    +(d_ij*d_kn+d_ik*d_jn+d_jk*d_in)*d_lo)/l6+
                    (d_in*(d_jo*dif[b3]*dif[b4]+d_ko*dif[b2]*dif[b4]+d_lo*dif[b2]*dif[b3])+
                   d_jn*(d_io*dif[b3]*dif[b4]+d_ko*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b3])+
                   d_kn*(d_io*dif[b2]*dif[b4]+d_jo*dif[b1]*dif[b4]+d_lo*dif[b1]*dif[b2])+
                   d_ln*(d_io*dif[b2]*dif[b3]+d_jo*dif[b1]*dif[b3]+d_ko*dif[b1]*dif[b2]))/l8))
            
            term4=((-dif[a1]/l2)*(
                    (d_in*(d_jo*(d_kp*dif[b4]+dif[b3]*d_lp)+d_ko*(d_jp*dif[b4]+dif[b2]*d_lp)+d_lo*(d_jp*dif[b3]+dif[b2]*d_kp))+
                   d_jn*(d_io*(d_kp*dif[b4]+dif[b3]*d_lp)+d_ko*(d_ip*dif[b4]+dif[b1]*d_lp)+d_lo*(d_ip*dif[b3]+dif[b1]*d_kp))+
                   d_kn*(d_io*(d_jp*dif[b4]+dif[b2]*d_lp)+d_jo*(d_ip*dif[b4]+dif[b1]*d_lp)+d_lo*(d_ip*dif[b2]+dif[b1]*d_jp))+
                   d_ln*(d_io*(d_jp*dif[b3]+dif[b2]*d_kp)+d_jo*(d_ip*dif[b3]+dif[b1]*d_kp)+d_ko*(d_ip*dif[b2]+dif[b1]*d_jp))
                   )/l8))
            
            term5=((d_im*(d_jn*(d_ko*d_lp+d_kp*d_lo)+d_kn*(d_jo*d_lp+d_jp*d_lo)+d_ln*(d_jo*d_kp+d_jp*d_ko))+
                   d_jm*(d_in*(d_ko*d_lp+d_kp*d_lo)+d_kn*(d_io*d_lp+d_ip*d_lo)+d_ln*(d_io*d_kp+d_ip*d_ko))+
                   d_km*(d_in*(d_jo*d_lp+d_jp*d_lo)+d_jn*(d_io*d_lp+d_ip*d_lo)+d_ln*(d_io*d_jp+d_ip*d_jo))+
                   d_lm*(d_in*(d_jo*d_kp+d_jp*d_ko)+d_jn*(d_io*d_kp+d_ip*d_ko)+d_kn*(d_io*d_jp+d_ip*d_jo))
                   )/l8)
            term =term1+term2+term3+term4+term5

            return term
        

        if isinstance(a_inp, PointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            term = pd4(b_inp.derivative_indices,dif=diff)
            return term * exp_term


        elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, PointMeasurement):
            term = pd4(a_inp.derivative_indices,dif=diff)
            return term * exp_term

        elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            term= dd4(a_inp.derivative_index,b_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
            term= -dd4(b_inp.derivative_index,a_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            term=d2d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            term=d2d4(b_inp.derivative_indices,a_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            term=d3d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            term=-d3d4(b_inp.derivative_indices,a_inp.derivative_indices,diff)
            return term * exp_term

        elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            term=d4d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term

        else:
            raise TypeError("Unsupported measurement type for Gaussian covariance.")
            
            
    def build_symmetric(self, M):
        N = len(M)
        output_matrix = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                output_matrix[i, j] = self(M[i], M[j])

        return output_matrix


    def build_test(self, te, tr):
        N = len(tr)
        M = len(te)
        output_matrix = np.zeros((M, N), dtype=np.float64)

        for i in range(M):
            for j in range(N):
                output_matrix[i, j] = self(te[i], tr[j])

        return output_matrix

        
# Alternate effort to implement Matern 5_2 (untested)
class MaternCovariance52_generic(AbstractCovarianceFunction):

    def __init__(self, length_scale):
        self.length_scale = float(length_scale)
        self.nu = 2.5
        self.sqrt5 = np.sqrt(5.0)
        self.five_thirds = 5.0 / 3.0

    def __call__(self, a: AbstractMeasurement, b: AbstractMeasurement):
        coord1 = a.get_coordinate()
        coord2 = b.get_coordinate()
        diff = coord1 - coord2
        dist = np.linalg.norm(diff)
        l = self.length_scale
        r = np.sqrt( np.sum((coord1 - coord2)**2/l**2))

        if dist < 1e-10:
            if isinstance(a, PointMeasurement) and isinstance(b, PointMeasurement):
                return 1.0
            elif isinstance(a, dPointMeasurement) and isinstance(b, dPointMeasurement):
                index_a = a.derivative_index
                index_b = b.derivative_index
                return 5 / (3 * self.length_scale**2) if index_a == index_b else 0.0
            elif isinstance(a, ddPointMeasurement) and isinstance(b, ddPointMeasurement):
                index_a1, index_a2 = a.derivative_indices
                index_b1, index_b2 = b.derivative_indices
                if index_a1 == index_b1 and index_a2 == index_b2:
                    return 25/(3* self.length_scale**4)
                else:
                    return 0.0
            else:
                return 0.0

        exp_term = np.exp(-self.sqrt5 * r)
        poly = (1 + self.sqrt5 * r + self.five_thirds * r**2)

        if isinstance(a, PointMeasurement) and isinstance(b, PointMeasurement):
            return poly * exp_term

        elif isinstance(a, PointMeasurement) and isinstance(b, dPointMeasurement):
            index_b = b.derivative_index
            return (coord1[index_b] - coord2[index_b]) / l**2 * self.five_thirds * (1 + self.sqrt5 * r) * exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, PointMeasurement):
            index_a = a.derivative_index
            return - (coord1[index_a] - coord2[index_a]) / l**2 * self.five_thirds * (1 + self.sqrt5 * r) * exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, dPointMeasurement):
            index_a = a.derivative_index
            index_b = b.derivative_index
            delta_ij = 1.0 if index_a == index_b else 0.0
            return -self.five_thirds * exp_term * (
                5 * (coord1[index_a] - coord2[index_a]) * (coord1[index_b] - coord2[index_b]) / l**4
                - delta_ij / l**2 * (1 + self.sqrt5 * r)
            )

        elif isinstance(a, PointMeasurement) and isinstance(b, ddPointMeasurement):
            index_b1, index_b2 = b.derivative_indices
            return - self.five_thirds * exp_term * (
                5 * (coord1[index_b1] - coord2[index_b1]) * (coord1[index_b2] - coord2[index_b2]) / l**4
                - (1 + self.sqrt5 * r) / l**2 * (1.0 if index_b1 == index_b2 else 0.0)
            )

        elif isinstance(a, ddPointMeasurement) and isinstance(b, PointMeasurement):
            index_a1, index_a2 = a.derivative_indices
            return - self.five_thirds * exp_term * (
                5 * (coord1[index_a1] - coord2[index_a1]) * (coord1[index_a2] - coord2[index_a2]) / l**4
                - (1 + self.sqrt5 * r) / l**2 * (1.0 if index_a1 == index_a2 else 0.0)
            )

        elif isinstance(a, dPointMeasurement) and isinstance(b, ddPointMeasurement):
            index_a = a.derivative_index
            index_b1, index_b2 = b.derivative_indices
            return - self.five_thirds * exp_term * (
                -5 * 5 * (coord1[index_a] - coord2[index_a]) * (coord1[index_b1] - coord2[index_b1]) * (coord1[index_b2] - coord2[index_b2]) / l**6
                + 5 * ((1 + self.sqrt5 * r) / l**4) * ((1.0 if index_a == index_b1 else 0.0) * (coord1[index_b2] - coord2[index_b2]) + (1.0 if index_a == index_b2 else 0.0) * (coord1[index_b1] - coord2[index_b1]))
            )


        elif isinstance(a, ddPointMeasurement) and isinstance(b, dPointMeasurement):
            index_a1, index_a2 = a.derivative_indices
            index_b = b.derivative_index
            return self.five_thirds * exp_term * (
                -5 * 5 * (coord1[index_a1] - coord2[index_a1]) * (coord1[index_a2] - coord2[index_a2]) * (coord1[index_b] - coord2[index_b]) / l**6
                + 5 * ((1 + self.sqrt5 * r) / l**4) * ((1.0 if index_b == index_a1 else 0.0) * (coord1[index_a2] - coord2[index_a2]) + (1.0 if index_b == index_a2 else 0.0) * (coord1[index_a1] - coord2[index_a1]))
            )


        elif isinstance(a, ddPointMeasurement) and isinstance(b, ddPointMeasurement):
            index_a1, index_a2 = a.derivative_indices
            index_b1, index_b2 = b.derivative_indices
            delta_a1b1 = 1.0 if index_a1 == index_b1 else 0.0
            delta_a2b2 = 1.0 if index_a2 == index_b2 else 0.0
            delta_a1b2 = 1.0 if index_a1 == index_b2 else 0.0
            delta_a2b1 = 1.0 if index_a2 == index_b1 else 0.0

            return self.five_thirds * exp_term * (
                25 * (coord1[index_a1] - coord2[index_a1]) * (coord1[index_a2] - coord2[index_a2]) * (coord1[index_b1] - coord2[index_b1]) * (coord1[index_b2] - coord2[index_b2]) / l**8
                - 5 * ((1 + self.sqrt5 * r) / l**6) * (
                    delta_a1b1 * (coord1[index_a2] - coord2[index_a2]) * (coord1[index_b2] - coord2[index_b2])
                    + delta_a1b2 * (coord1[index_a2] - coord2[index_a2]) * (coord1[index_b1] - coord2[index_b1])
                    + delta_a2b1 * (coord1[index_a1] - coord2[index_a1]) * (coord1[index_b2] - coord2[index_b2])
                    + delta_a2b2 * (coord1[index_a1] - coord2[index_a1]) * (coord1[index_b1] - coord2[index_b1])
                )
                + (1 + self.sqrt5 * r) / l**4 * (delta_a1b1 * delta_a2b2 + delta_a1b2 * delta_a2b1)
            )


        else:
            raise TypeError("Unsupported measurement type for Matern (nu=5/2) covariance.")
        
# Gaussian kernel specific to 1D (tested for various functions)

class GaussianCovariance_generic_1D(AbstractCovarianceFunction):

    def __init__(self, length_scale):
        self.length_scale = length_scale

    def __call__(self, a: AbstractMeasurement, b: AbstractMeasurement):
        """Compute the Gaussian covariance between two measurement points."""
        x = a.coordinate
        y = b.coordinate
        dist_sq = np.sum((x - y)**2)
        l2 = self.length_scale**2
        l4 = self.length_scale**4
        l6 = self.length_scale**6
        l8 = self.length_scale**8
        l10 = self.length_scale**10
        l12 = self.length_scale**12
        l14 = self.length_scale**14
        l16 = self.length_scale**16
        exp_term = np.exp(-0.5 * dist_sq / l2)

        if isinstance(a, PointMeasurement) and isinstance(b, PointMeasurement):
            return exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, dPointMeasurement):
            term = (1/l2 - (x-y)**2 / l4)
            return term * exp_term


        elif isinstance(a, ddPointMeasurement) and isinstance(b, ddPointMeasurement):
            term_val=3/l4-(6*(x-y)**2)/l6+(x-y)**4/l8

            return term_val * exp_term

        elif (isinstance(a, PointMeasurement) and isinstance(b, dPointMeasurement)):
            return ((x-y) / l2) * exp_term

        elif (isinstance(a, dPointMeasurement) and isinstance(b, PointMeasurement)):
            return ((y-x) / l2) * exp_term

        elif isinstance(a, PointMeasurement) and isinstance(b, ddPointMeasurement):
            term = (-1/l2 + (x-y)**2 / l4)

            return term * exp_term

        elif isinstance(a, ddPointMeasurement) and isinstance(b, PointMeasurement):
            term=(-1/l2 + (x-y)**2 / l4)
            return term * exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, ddPointMeasurement):
            term_val=(3*(x-y)/l4-((x-y)**3)/l6)
            return term_val * exp_term

        elif isinstance(a, ddPointMeasurement) and isinstance(b, dPointMeasurement):
            term_val=(-3*(x-y)/l4+((x-y)**3)/l6)
            return term_val * exp_term

        # Handling dddPointMeasurement
        elif isinstance(a, PointMeasurement) and isinstance(b, dddPointMeasurement):
            term = (-3*(x-y)/l4+((x-y)**3)/l6)
            return term * exp_term

        elif isinstance(a, dddPointMeasurement) and isinstance(b, PointMeasurement):
            term=(3*(x-y)/l4-((x-y)**3)/l6)
            return term * exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, dddPointMeasurement):
            term =(-3/l4+(6*(x-y)**2)/l6-(x-y)**4/l8)
            return term * exp_term

        elif isinstance(a, dddPointMeasurement) and isinstance(b, dPointMeasurement):
            term =(-3/l4+(6*(x-y)**2)/l6-(x-y)**4/l8)
            return term * exp_term

        elif isinstance(a, ddPointMeasurement) and isinstance(b, dddPointMeasurement):
            term=(15*(x-y)/l6)-(10*(x-y)**3/l8)+((x-y)**5)/l10
            return term * exp_term

        elif isinstance(a, dddPointMeasurement) and isinstance(b, ddPointMeasurement):
            term=(-15*(x-y)/l6)+(10*(x-y)**3/l8)-((x-y)**5)/l10
            return term * exp_term

        
        elif isinstance(a, dddPointMeasurement) and isinstance(b, dddPointMeasurement):
            term= 15/l6-(45*(x-y)**2/l8)+15*((x-y)**4)/l10-(x-y)**6/l12
            return term * exp_term

        # Handling ddddPointMeasurement 
        elif isinstance(a, PointMeasurement) and isinstance(b, ddddPointMeasurement):
            term=3/l4-(6*(x-y)**2)/l6+((x-y)**4)/l8
            return term * exp_term

        elif isinstance(a, ddddPointMeasurement) and isinstance(b, PointMeasurement):
            term=3/l4-(6*(x-y)**2)/l6+((x-y)**4)/l8
            return term * exp_term

        elif isinstance(a, dPointMeasurement) and isinstance(b, ddddPointMeasurement):
            term=(-15*(x-y))/l6+(10*(x-y)**3)/l8-((x-y)**5)/l10
            return term * exp_term

        elif isinstance(a, ddddPointMeasurement) and isinstance(b, dPointMeasurement):
            term = (-15*(x-y))/l6+(10*(x-y)**3)/l8-((x-y)**5)/l10
            term=-term
            return term * exp_term

        elif isinstance(a, ddPointMeasurement) and isinstance(b, ddddPointMeasurement):
            term=-15/l6+(45*(x-y)**2)/l8-(15*(x-y)**4)/l10+((x-y)**6)/l12
            return term * exp_term

        elif isinstance(a, ddddPointMeasurement) and isinstance(b, ddPointMeasurement):
            term=-15/l6+(45*(x-y)**2)/l8-(15*(x-y)**4)/l10+((x-y)**6)/l12
            return term * exp_term

        elif isinstance(a, dddPointMeasurement) and isinstance(b, ddddPointMeasurement):
            term = (105*(x-y))/l8-(105*(x-y)**3)/l10+(21*(x-y)**5)/l12-((x-y)**7)/l14
            return term * exp_term
        
        elif isinstance(a, ddddPointMeasurement) and isinstance(b, dddPointMeasurement):
            term = (105*(x-y))/l8-(105*(x-y)**3)/l10+(21*(x-y)**5)/l12-((x-y)**7)/l14
            term=-term
            return term * exp_term

        elif isinstance(a, ddddPointMeasurement) and isinstance(b, ddddPointMeasurement):
            term = (105/l8)-(420*(x-y)**2)/l10+(210*(x-y)**4)/l12-(28*(x-y)**6)/l14+((x-y)**8)/l16
            return term * exp_term


        else:
            raise TypeError("Unsupported measurement type for Gaussian covariance.")
        
        