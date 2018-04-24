using ForwardDiff
using IterativeSolvers
using Plots

function rosenbrock(x)
    a = 1.0
    b = 100.0
    [(a-x[1])^2 + b*(x[2] - x[1]^2)^2]
end
 
function square(x)
    [x[1]*x[1]*x[1]*x[1] - 612]
end

function mysin(x)
    [sin(x[1])]
end

function newton(x0, f, eps = 1e-6, maxiter = 1000, debug = false)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    x = x0
    iter = 0
    while norm(J(x)) > eps  && iter < maxiter
        hes = H(x)
        jac = J(x)
        s = hes\transpose(jac)
        x = x - s
        debug && println("x: ", x, " ", s)
        iter=iter+1
    end
    x, iter, norm(J(x))
end

function halley(x0, f, eps = 1e-6, maxiter = 1000, debug = false)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    T = x -> ForwardDiff.jacobian(H, x);
    x = x0
    iter = 0
    dim = size(x0,1)
    while norm(J(x)) > eps  && iter < maxiter
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
        b = 2*jac*hes
        s = A\transpose(b)
        x = x - s
        debug && println("x: ", x, " ", s)
        iter=iter+1
    end
    x, iter, norm(J(x))
end

function cheby(x0, f, eps = 1e-6, maxiter = 1000, debug = false)
    J = x -> ForwardDiff.jacobian(f, x);
    H = x -> ForwardDiff.jacobian(J, x);
    T = x -> ForwardDiff.jacobian(H, x);
    x = x0
    iter = 0
    dim = size(x0,1)
    while norm(J(x)) > eps  && iter < maxiter
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
        A = hes
        s3 = zeros(s)
        for i = 1:dim 
          s3[i] = 0
          for j = 1:dim 
              s3[i] += contr[i,j] * s[j]
          end
        end
        b = 0.5*s3
        s2 = A\b

        x = x - s - s2
        debug && println("x: ", x, " ", s, " ", s2)
        iter=iter+1
    end
    x, iter, norm(J(x))
end
x2 = [-1.0]
x0 = [-3.0, -4.0]
x1 = [10.0]

x = x1 ; f = square
x = x0 ; f = rosenbrock
x = x2 ; f = mysin
debug = false
myeps = 1e-6
maxiter = 20

funclist = [[x0, rosenbrock], [x1, square], [x2, mysin]] 

for x in funclist
    println("Running ", x[2], " with starting point ", x[1], ".")
    try 
        res = newton(x[1], x[2], myeps, maxiter, debug)
        println("Newton solution ", res[1], " with ", res[2], " iterations and residual norm(J) of ", res[3], ".")
    catch ex
        println("Newton not converged with exception ", ex, ".")
        debug && throw(ex)
    end
    
    try 
        res = halley(x[1], x[2], myeps, maxiter, debug)
        println("Halley solution ", res[1], " with ", res[2], " iterations and residual norm(J) of ", res[3], ".")
    catch ex
        println("Halley not converged with exception ", ex, ".")
        debug && throw(ex)
    end
    
    
    try 
        res = cheby(x[1], x[2], myeps, maxiter, debug)
        println("Chebychev solution ", res[1], " with ", res[2], " iterations and residual norm(J) of ", res[3], ".")
    catch ex
        println("Chebychev not converged with exception ", ex, ".")
        debug && throw(ex)
    end
    println("")
end
    




