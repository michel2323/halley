using ForwardDiff
using IterativeSolvers

function rosenbrock(x)
    a = 1.0
    b = 100.0
    [(a-x[1])^2 + b*(x[2] - x[1]^2)^2]
end
 
function square(x)
    return [x[1]*x[1]*x[1]*x[1] - 612]
end

function mysin(x)
    return [sin(x[1])]
end

# function myfunc(x)
#     cos(-((x[1]-3)^2+(x[2]-3)^2))
# end

function newton(x0, f)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    eps = 1e-6
    x = x0
    iter = 0
    maxiter = 1000
    while norm(J(x)) > eps  && iter < maxiter
        # println("Newton iteration: ", iter)
        # println("J: ", J(x))
        # println("f: ", f(x))
        hes = H(x)
        jac = J(x)
        s = hes\transpose(jac)
        x = x - s
        println("x: ", x, " ", s)
        iter=iter+1
    end
    println("Gradient: ", J(x))
    x, iter
end

function halley(x0, f)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    T = x -> ForwardDiff.jacobian(H, x);
    # T = x -> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x);
    eps = 1e-6
    x = x0
    iter = 0
    maxiter = 20
    dim = size(x0,1)
    while norm(J(x)) > eps  && iter < maxiter
        println("Halley iteration: ", iter)
        ten = T(x)
        hes = H(x)
        jac = J(x)
        ten = reshape(ten,dim,dim,dim)
        contr = zeros(hes)
        for i = 1:dim 
          for j = 1:dim 
            contr[i,j] = 0.0
            for k = 1:dim 
              contr[i,j] += ten[i,j,k] * jac[k]
            end
          end
        end
        A = 2*hes^2 - transpose(contr)
        println("A: ", A, " hes: ", hes^2, " contr: ", contr)
        b = 2*jac*hes
        s = A\transpose(b)
        x = x - s
        println("x: ", x, " ", s)
        iter=iter+1
    end
    println("Gradient: ", norm(J(x)))
    println("Hessian: ", norm(H(x)))
    println("Tensor: ", norm(T(x)))
    x, iter
end

function halley2(x0, f)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    T = x -> ForwardDiff.jacobian(H, x);
    # T = x -> ForwardDiff.jacobian(x -> ForwardDiff.jacobian(f, x), x);
    eps = 1e-6
    x = x0
    iter = 0
    maxiter = 20
    dim = size(x0,1)
    while norm(J(x)) > eps  && iter < maxiter
        println("Halley iteration: ", iter)
        ten = T(x)
        hes = H(x)
        jac = J(x)
        s = hes\transpose(jac)
        ten = reshape(ten,dim,dim,dim)
        contr = zeros(hes)
        for i = 1:dim 
          for j = 1:dim 
            contr[i,j] = 0.0
            for k = 1:dim 
              contr[i,j] += ten[i,j,k] * s[k]
            end
          end
        end
        A = hes + 0.5*transpose(contr)
        A = hes
        println("s: ", s)
        println("contr: ", contr)
        # println("s*s: ", transpose(s)*s)
        b = 0.5*contr*s
        s2 = A\b

        x = x - s - s2
        println("x: ", x, " ", s, " ", s2)
        println("Tensor: ", norm(T(x)))
        iter=iter+1
    end
    println("Gradient: ", norm(J(x)))
    println("Hessian: ", norm(H(x)))
    println("Tensor: ", norm(T(x)))
    x, iter
end
x2 = [-1.0]
x0 = [-3.0, -4.0]
x1 = [10.0]
x3 = [3.14,3.14]

println("Solution at ", newton(x0, rosenbrock))
println("Solution at ", halley(x0, rosenbrock))
# println("Solution: ", newton(x1, square))
# println("Solution at ", halley(x1, square))

# println("Solution: ", newton(x2, mysin))
# println("Solution at ", halley(x2, mysin))

# println(myfunc(x3))
# println("Solution: ", newton(x3, myfunc))
# println("Solution at ", halley(x3, myfunc))


