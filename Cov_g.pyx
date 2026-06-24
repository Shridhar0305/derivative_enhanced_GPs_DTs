
import numpy as np
# Import the C-level NumPy API
cimport numpy as cnp
# Import compiler directives
cimport cython
# Import the C math library's exp function for speed
from libc.math cimport exp

# Initialize the NumPy C-API (required)
cnp.import_array()

# Decorators to disable Python's safety checks for more speed
@cython.boundscheck(False)
@cython.wraparound(False)
cdef class gaussian_cov_generic:
    # Declare instance variables with C types
    cdef double length_scale
    cdef object _diff_buffer

    cdef double pd4(self, object ind, object dif_obj, double ls):
            cdef int b1, b2, b3, b4
            cdef double term
            # Cast the object to a typed memoryview for C-level access
            cdef cnp.ndarray[cnp.double_t, ndim=1] dif = dif_obj
            
            b1, b2, b3, b4 = ind

            cdef double l2,l4,l6,l8,l10
            l2=ls**2
            l4=ls**4
            l6=ls**6
            l8=ls**8
            l10=ls**10


            cdef int d_ij=1 if b1==b2 else 0
            cdef int d_kl=1 if b3==b4 else 0
            cdef int d_ik=1 if b1==b3 else 0
            cdef int d_jl=1 if b2==b4 else 0
            cdef int d_jk=1 if b2==b3 else 0
            cdef int d_il=1 if b1==b4 else 0


            term = (d_ij*d_kl+d_ik*d_jl+d_jk*d_il)/l4-((d_ik*dif[b2]+d_jk*dif[b1]+d_ij*dif[b3])*dif[b4]/l6)-(
                    (d_il*dif[b2]+d_jl*dif[b1])*dif[b3]/l6)-(d_kl*dif[b1]*dif[b2])/l6+(dif[b1]*dif[b2]*dif[b3]*dif[b4])/l8
            return term
        
    cdef double dd4(self, object ind1, object ind2, object dif_obj, double ls):
            cdef int a1,b1, b2, b3, b4
            a1 =ind1
            b1, b2, b3, b4=ind2
            # Cast the object to a typed memoryview for C-level access
            cdef cnp.ndarray[cnp.double_t, ndim=1] dif = dif_obj

            cdef double l2,l4,l6,l8,l10
            l2=ls**2
            l4=ls**4
            l6=ls**6
            l8=ls**8
            l10=ls**10

            cdef int d_ij=1 if b1==b2 else 0
            cdef int d_kl=1 if b3==b4 else 0
            cdef int d_ik=1 if b1==b3 else 0
            cdef int d_jl=1 if b2==b4 else 0
            cdef int d_jk=1 if b2==b3 else 0
            cdef int d_il=1 if b1==b4 else 0
            cdef int d_im=1 if a1==b1 else 0
            cdef int d_jm=1 if a1==b2 else 0
            cdef int d_km=1 if a1==b3 else 0
            cdef int d_lm=1 if a1==b4 else 0

            cdef double term1, term2, term3, term

            term1=-(dif[a1]/l2)*self.pd4(ind2, dif_obj=dif, ls=ls)
            term2=-((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*dif[b1]+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*dif[b2]+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*dif[b3]
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*dif[b4])/l6
            term3=(d_im*dif[b2]*dif[b3]*dif[b4]+d_jm*dif[b1]*dif[b3]*dif[b4]
                    +d_km*dif[b1]*dif[b2]*dif[b4]+d_lm*dif[b1]*dif[b2]*dif[b3])/l8
            term = term1 +term2+term3
            return term
        
    cdef double d2d4(self, object ind1, object ind2, object dif_obj, double ls):
            cdef int a1,a2,b1, b2, b3, b4
            a1,a2=ind1
            b1, b2, b3, b4=ind2
            # Cast the object to a typed memoryview for C-level access
            cdef cnp.ndarray[cnp.double_t, ndim=1] dif = dif_obj

            cdef double l2,l4,l6,l8,l10
            l2=ls**2
            l4=ls**4
            l6=ls**6
            l8=ls**8
            l10=ls**10

            cdef int d_ij=1 if b1==b2 else 0
            cdef int d_kl=1 if b3==b4 else 0
            cdef int d_ik=1 if b1==b3 else 0
            cdef int d_jl=1 if b2==b4 else 0
            cdef int d_jk=1 if b2==b3 else 0
            cdef int d_il=1 if b1==b4 else 0
            cdef int d_im=1 if a1==b1 else 0
            cdef int d_jm=1 if a1==b2 else 0
            cdef int d_km=1 if a1==b3 else 0
            cdef int d_lm=1 if a1==b4 else 0
            cdef int d_in=1 if a2==b1 else 0
            cdef int d_jn=1 if a2==b2 else 0
            cdef int d_kn=1 if a2==b3 else 0
            cdef int d_ln=1 if a2==b4 else 0
            cdef int d_mn=1 if a1==a2 else 0
            cdef double term1, term2, term3, term4, term


            term1=(-dif[a2]*self.dd4(a1,ind2,dif,ls)/l2)-self.pd4(ind2,dif,ls)*d_mn/l2
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

    cdef double d3d4(self, object ind1, object ind2, object dif_obj, double ls):
            cdef int a1,a2,a3,b1, b2, b3, b4
            a1,a2,a3=ind1
            b1, b2, b3, b4=ind2
            # Cast the object to a typed memoryview for C-level access
            cdef cnp.ndarray[cnp.double_t, ndim=1] dif = dif_obj

            cdef double l2,l4,l6,l8,l10
            l2=ls**2
            l4=ls**4
            l6=ls**6
            l8=ls**8
            l10=ls**10

            cdef int d_ij=1 if b1==b2 else 0
            cdef int d_kl=1 if b3==b4 else 0
            cdef int d_ik=1 if b1==b3 else 0
            cdef int d_jl=1 if b2==b4 else 0
            cdef int d_jk=1 if b2==b3 else 0
            cdef int d_il=1 if b1==b4 else 0
            cdef int d_im=1 if a1==b1 else 0
            cdef int d_jm=1 if a1==b2 else 0
            cdef int d_km=1 if a1==b3 else 0
            cdef int d_lm=1 if a1==b4 else 0
            cdef int d_in=1 if a2==b1 else 0
            cdef int d_jn=1 if a2==b2 else 0
            cdef int d_kn=1 if a2==b3 else 0
            cdef int d_ln=1 if a2==b4 else 0
            cdef int d_mn=1 if a1==a2 else 0
            cdef int d_io=1 if a3==b1 else 0
            cdef int d_jo=1 if a3==b2 else 0
            cdef int d_ko=1 if a3==b3 else 0
            cdef int d_lo=1 if a3==b4 else 0
            cdef int d_mo=1 if a3==a1 else 0
            cdef int d_no=1 if a3==a2 else 0
            cdef double term1, term2, term3, term4, term5, term6, term


            term1=(-dif[a3]*self.d2d4(ind1=[a1,a2], ind2=ind2, dif_obj=dif, ls=ls)/l2)-(d_no*self.dd4(a1,ind2,dif,ls)/l2)
            term2=(-dif[a2]/l2)*((-self.pd4(ind2,dif,ls)*d_mo/l2)+(-dif[a1]/l2)*(-1*((d_jl*d_ko+d_kl*d_jo+d_jk*d_lo)*dif[b1]+(d_il*d_ko+d_kl*d_io+d_ik*d_lo)*dif[b2]
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
        
    cdef double d4d4(self, object ind1, object ind2, object dif_obj, double ls):
            cdef int a1,a2,a3,a4,b1, b2, b3, b4
            a1,a2,a3,a4=ind1
            b1, b2, b3, b4=ind2
            # Cast the object to a typed memoryview for C-level access
            cdef cnp.ndarray[cnp.double_t, ndim=1] dif = dif_obj

            cdef double l2,l4,l6,l8,l10,l12
            l2=ls**2
            l4=ls**4
            l6=ls**6
            l8=ls**8
            l10=ls**10
            l12=ls**12

            cdef int d_ij=1 if b1==b2 else 0
            cdef int d_kl=1 if b3==b4 else 0
            cdef int d_ik=1 if b1==b3 else 0
            cdef int d_jl=1 if b2==b4 else 0
            cdef int d_jk=1 if b2==b3 else 0
            cdef int d_il=1 if b1==b4 else 0
            cdef int d_im=1 if a1==b1 else 0
            cdef int d_jm=1 if a1==b2 else 0
            cdef int d_km=1 if a1==b3 else 0
            cdef int d_lm=1 if a1==b4 else 0
            cdef int d_in=1 if a2==b1 else 0
            cdef int d_jn=1 if a2==b2 else 0
            cdef int d_kn=1 if a2==b3 else 0
            cdef int d_ln=1 if a2==b4 else 0
            cdef int d_mn=1 if a1==a2 else 0
            cdef int d_io=1 if a3==b1 else 0
            cdef int d_jo=1 if a3==b2 else 0
            cdef int d_ko=1 if a3==b3 else 0
            cdef int d_lo=1 if a3==b4 else 0
            cdef int d_mo=1 if a3==a1 else 0
            cdef int d_no=1 if a3==a2 else 0
            cdef int d_ip=1 if a4==b1 else 0
            cdef int d_jp=1 if a4==b2 else 0
            cdef int d_kp=1 if a4==b3 else 0
            cdef int d_lp=1 if a4==b4 else 0
            cdef int d_mp=1 if a4==a1 else 0
            cdef int d_np=1 if a4==a2 else 0
            cdef int d_po=1 if a3==a4 else 0
            cdef double pd4_xp,pd4_xo,dd4_xp,dd4_xo,pd4_xo_xp,dd4_xo_xp,d2d4_xp
            cdef double term1,term2,term3,term4,term5,term

            pd4_xp=(-1*((d_jl*d_kp+d_kl*d_jp+d_jk*d_lp)*dif[b1]+(d_il*d_kp+d_kl*d_ip+d_ik*d_lp)*dif[b2]
                    +(d_il*d_jp+d_jl*d_ip+d_ij*d_lp)*dif[b3]+(d_ij*d_kp+d_ik*d_jp+d_jk*d_ip)*dif[b4])/l6+
                    ((d_ip*dif[b2]*dif[b3]*dif[b4])+(d_jp*dif[b1]*dif[b3]*dif[b4])+(d_kp*dif[b1]*dif[b2]*dif[b4])+
                     (d_lp*dif[b1]*dif[b2]*dif[b3]))/l8)
            
            pd4_xo=(-1*((d_jl*d_ko+d_kl*d_jo+d_jk*d_lo)*dif[b1]+(d_il*d_ko+d_kl*d_io+d_ik*d_lo)*dif[b2]
                    +(d_il*d_jo+d_jl*d_io+d_ij*d_lo)*dif[b3]+(d_ij*d_ko+d_ik*d_jo+d_jk*d_io)*dif[b4])/l6+
                    ((d_io*dif[b2]*dif[b3]*dif[b4])+(d_jo*dif[b1]*dif[b3]*dif[b4])+(d_ko*dif[b1]*dif[b2]*dif[b4])+
                     (d_lo*dif[b1]*dif[b2]*dif[b3]))/l8)
            
            dd4_xp=((-self.pd4(ind2,dif,ls)*d_mp/l2)+(-dif[a1]/l2)*(pd4_xp)-
                    ((d_jl*d_km+d_kl*d_jm+d_jk*d_lm)*d_ip+(d_il*d_km+d_kl*d_im+d_ik*d_lm)*d_jp+(d_il*d_jm+d_jl*d_im+d_ij*d_lm)*d_kp
                    +(d_ij*d_km+d_ik*d_jm+d_jk*d_im)*d_lp)/l6+
                    (d_im*(d_jp*dif[b3]*dif[b4]+d_kp*dif[b2]*dif[b4]+d_lp*dif[b2]*dif[b3])+
                   d_jm*(d_ip*dif[b3]*dif[b4]+d_kp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b3])+
                   d_km*(d_ip*dif[b2]*dif[b4]+d_jp*dif[b1]*dif[b4]+d_lp*dif[b1]*dif[b2])+
                   d_lm*(d_ip*dif[b2]*dif[b3]+d_jp*dif[b1]*dif[b3]+d_kp*dif[b1]*dif[b2])
                   )/l8)
            dd4_xo=((-self.pd4(ind2,dif,ls)*d_mo/l2)+(-dif[a1]/l2)*(pd4_xo)-
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


            d2d4_xp=(-self.dd4(a1,ind2,dif,ls)*(d_np/l2))-((dif[a2]/l2)*dd4_xp)-((d_mn/l2)*pd4_xp)
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
            
            term1=((-self.d3d4(ind1=[a1,a2,a3], ind2=ind2, dif_obj=dif, ls=ls)*(dif[a4]/l2))-((d_po/l2)*self.d2d4(ind1=[a1,a2], ind2=ind2, dif_obj=dif, ls=ls))-
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

    # The __init__ method is called from Python to create the object
    def __init__(self, double length_scale):
        self.length_scale = length_scale
        self._diff_buffer = None

    def build_symmetric(self,list M):

        cdef int N = len(M)
        cdef int i, j
        cdef double value
        cdef double[:, ::1] output_matrix = np.zeros((N, N), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                value = self(M[i], M[j])

                output_matrix[i, j] = value
        return output_matrix

    def build_test(self,list te,list tr):

        cdef int N = len(tr)
        cdef int M = len(te)
        cdef int i, j
        cdef double value
        cdef double[:, ::1] output_matrix = np.zeros((M, N), dtype=np.float64)

        for i in range(M):
            for j in range(N):
                value = self(te[i], tr[j])

                output_matrix[i, j] = value
        return output_matrix




    # 'cpdef' makes the method callable from both Python and other Cython code
    def __call__(self, dict a_inp, dict b_inp):
        # Extract and type NumPy arrays for efficient C-level access
        cdef cnp.ndarray[cnp.double_t, ndim=1] x = a_inp['cord']
        cdef cnp.ndarray[cnp.double_t, ndim=1] y = b_inp['cord']
        cdef int a_inp_len = a_inp['len']
        cdef int b_inp_len = b_inp['len']

        # Declare all local variables as C types
        cdef int d_dim = x.shape[0]
        cdef int i
        cdef double dist_sq = 0.0
        cdef double l2, l4, exp_term
        cdef double l6, l8, l10, l12
        cdef cnp.ndarray[cnp.double_t, ndim=1] diff
        cdef double term_val,term,t1,t2,t3,t4
        cdef int index_a,index_b,index_a1,index_a2,index_b1,index_b2,index_b3
        cdef int a1,a2,a3,b1,b2,b3
        cdef int delta_ik,delta_ij,delta_il,delta_jk,delta_jl,delta_kl
        cdef int d_ik,d_jk,d_ij,d_kl,d_il,d_jl,d_ip,d_jp,d_kp,d_lp
        cdef int d_iq,d_jq,d_kq,d_pq,d_lq,d_im,d_jm,d_km,d_lm
        cdef int d_in,d_jn,d_kn,d_ln,d_mn,d_io,d_jo,d_ko,d_lo,d_mo,d_no
        cdef int d_mp,d_np,d_po
        cdef object ind
        cdef double dif       
         
        
        

        # Perform calculations at C-level
        # This loop is much faster than the pure Python/NumPy version
        for i in range(d_dim):
            dist_sq += (x[i] - y[i]) * (x[i] - y[i])

        # l2 = self.length_scale * self.length_scale
        # l4 = l2 * l2
        l2 = self.length_scale**2
        l4 = self.length_scale**4
        l6 = self.length_scale**6
        l8 = self.length_scale**8
        l10 = self.length_scale**10
        l12 = self.length_scale**12
        exp_term = exp(-0.5 * dist_sq / l2)

        # Logic for returning the correct covariance value
        if a_inp_len == 0 and b_inp_len == 0:
            return exp_term

        # For derivative cases, calculate the difference vector using NumPy
        # Re-use a buffer to avoid repeated allocations. High-frequency allocations
        # inside a loop can appear as a memory leak if the garbage collector
        # does not run fast enough to reclaim the memory.
        if self._diff_buffer is None or self._diff_buffer.shape[0] != d_dim:
            self._diff_buffer = np.empty(d_dim, dtype=np.float64)

        np.subtract(x, y, out=self._diff_buffer)
        diff = self._diff_buffer

        if a_inp_len==0 and b_inp_len==0:
            return exp_term
        elif a_inp_len==1 and b_inp_len==1:
            index_a = a_inp['index']
            index_b = b_inp['index']

            if index_a == index_b:
                return (1/l2 - diff[index_a] * diff[index_b] / l4) * exp_term
            else:
                return (- diff[index_a] * diff[index_b] / l4) * exp_term

        elif a_inp_len==2 and b_inp_len==2:
            index_a1, index_a2 = a_inp['index']
            index_b1, index_b2 = b_inp['index']

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

        elif a_inp_len==0 and b_inp_len==1:
            index_b = b_inp['index']
            return ((diff[index_b]) / l2) * exp_term

        elif a_inp_len==1 and b_inp_len==0:
            index_a = a_inp['index']
            return (-(diff[index_a]) / l2) * exp_term

        elif a_inp_len==0 and b_inp_len==2:
            index_b1, index_b2 = b_inp['index']
            d_ij=1 if index_b1==index_b2 else 0
            term = (-d_ij/l2 + (diff[index_b1]*diff[index_b2]) / l4)

            return term * exp_term

        elif a_inp_len==2 and b_inp_len==0:
            index_a1, index_a2 = a_inp['index']
            d_ij=1 if index_a1==index_a2 else 0
            term = (-d_ij/l2 + (diff[index_a1]*diff[index_a2]) / l4)
            return term * exp_term

        elif a_inp_len==1 and b_inp_len==2:
            index_a=a_inp['index']
            index_b1, index_b2 = b_inp['index']

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
        
        elif a_inp_len==2 and b_inp_len==1:
            index_a1, index_a2 = a_inp['index']
            index_b = b_inp['index']

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
        elif a_inp_len==0 and b_inp_len==3:
            index_b1, index_b2, index_b3 = b_inp['index']
            
        # elif isinstance(a_inp, PointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            delta_ik = 1 if index_b1 == index_b2 else 0
            delta_jk = 1 if index_b1 == index_b3 else 0
            delta_ij = 1 if index_b2 == index_b3 else 0
            term = (diff[index_b1] * diff[index_b2] * diff[index_b3] / l6)
            term+=(-(delta_ik * diff[index_b3] +
                delta_jk * diff[index_b2] +
                delta_ij * diff[index_b1]) / l4)

            return term * exp_term

        elif a_inp_len==3 and b_inp_len==0:
            index_a1, index_a2, index_a3 = a_inp['index']
            
        # elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, PointMeasurement):
            # index_a1, index_a2, index_a3 = a_inp.derivative_indices
            delta_ik = 1 if index_a1 == index_a2 else 0
            delta_jk = 1 if index_a1 == index_a3 else 0
            delta_ij = 1 if index_a2 == index_a3 else 0

            term = (-diff[index_a1] * diff[index_a2] * diff[index_a3] / l6)
            term+=((delta_ik * diff[index_a3] +
                delta_jk * diff[index_a2] +
                delta_ij * diff[index_a1]) / l4)

            return term * exp_term
        

        elif a_inp_len==1 and b_inp_len==3:
            a=a_inp['index']
            b1, b2, b3 = b_inp['index']
        # elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            # a = a_inp.derivative_index
            # b1, b2, b3 = b_inp.derivative_indices
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


        elif a_inp_len==3 and b_inp_len==1:
            a=b_inp['index']
            b1, b2, b3 = a_inp['index']
        
        # elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
        #     a = b_inp.derivative_index
        #     b1, b2, b3 = a_inp.derivative_indices
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
        

        elif a_inp_len==2 and b_inp_len==3:
        # elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            a1, a2= a_inp['index']
            b1, b2, b3 = b_inp['index']

            # a1, a2 = a_inp.derivative_indices
            # b1, b2, b3 = b_inp.derivative_indices
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


        elif a_inp_len==3 and b_inp_len==2:
        # elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            a1,a2=b_inp['index']
            b1, b2, b3 = a_inp['index']

            # a1, a2 = b_inp.derivative_indices
            # b1, b2, b3 = a_inp.derivative_indices
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

        elif a_inp_len==3 and b_inp_len==3:
        # elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
            a1, a2, a3 = a_inp['index']
            b1, b2, b3 = b_inp['index']

            # a1, a2, a3 = a_inp.derivative_indices
            # b1, b2, b3 = b_inp.derivative_indices

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
        

        elif a_inp_len==0 and b_inp_len==4:
            term=self.pd4(b_inp['index'], dif_obj=diff, ls=self.length_scale)
        # if isinstance(a_inp, PointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
        #     term = pd4(b_inp.derivative_indices,dif=diff)
            return term * exp_term

        elif a_inp_len==4 and b_inp_len==0:
            term = self.pd4(a_inp['index'], dif_obj=diff, ls=self.length_scale)
        # elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, PointMeasurement):
            # term = pd4(a_inp.derivative_indices,dif=diff)
            return term * exp_term

        elif a_inp_len==1 and b_inp_len==4:
            term=self.dd4(a_inp['index'],b_inp['index'],diff,ls=self.length_scale)
        # elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            # term= dd4(a_inp.derivative_index,b_inp.derivative_indices,diff)
            return term * exp_term

        elif a_inp_len==4 and b_inp_len==1:
            term=-self.dd4(b_inp['index'],a_inp['index'],diff,ls=self.length_scale)
        #
        # elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
            # term= -dd4(b_inp.derivative_index,a_inp.derivative_indices,diff)
            return term * exp_term

        elif a_inp_len==2 and b_inp_len==4:
            term=self.d2d4(a_inp['index'],b_inp['index'],diff,ls=self.length_scale)
        #
        # elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
        #     term=d2d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term

        elif a_inp_len==4 and b_inp_len==2:
            term=self.d2d4(b_inp['index'],a_inp['index'],diff,ls=self.length_scale)
        #
        # elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
            # term=d2d4(b_inp.derivative_indices,a_inp.derivative_indices,diff)
            return term * exp_term

        elif a_inp_len==3 and b_inp_len==4:
            term=self.d3d4(a_inp['index'],b_inp['index'],diff,ls=self.length_scale)
        #
        #
        # elif isinstance(a_inp, dddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            # term=d3d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term


        elif a_inp_len==4 and b_inp_len==3:
            term=-self.d3d4(b_inp['index'],a_inp['index'],diff,ls=self.length_scale)
        #
        # #
        # elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, dddPointMeasurement):
        #     term=-d3d4(b_inp.derivative_indices,a_inp.derivative_indices,diff)
            return term * exp_term

        elif a_inp_len==4 and b_inp_len==4:
            term=self.d4d4(a_inp['index'],b_inp['index'],diff,ls=self.length_scale)
        #
        #
        # elif isinstance(a_inp, ddddPointMeasurement) and isinstance(b_inp, ddddPointMeasurement):
            # term=d4d4(a_inp.derivative_indices,b_inp.derivative_indices,diff)
            return term * exp_term

        else:
            raise TypeError("Unsupported measurement type for Gaussian covariance.")