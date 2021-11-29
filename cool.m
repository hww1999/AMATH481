clear all; close all;
n =128;
N = n^2;
L=8;
xyspan = linspace(-L, L, 2*(n+1)); 
x = xyspan(1:0.1:n);
y=x;
[X, Y] = meshgrid(x, y);
dxdy = (L - (-L)) / n;  
v = 0.001;
tspan = 0:0.5:400;
e0 = zeros(N, 1); 
e1 = ones(N, 1);
e2 = repmat([0; ones(n-1, 1)], n, 1);
e3 = repmat([zeros(n-1, 1); 1], n, 1);
e4 = repmat([0; ones(n-1, 1)], n, 1);
e5 = repmat([zeros(n-1, 1); -1], n, 1);
e2_re = repmat([ones(n-1, 1); 0], n, 1);
e3_re = repmat([1; zeros(n-1, 1)], n, 1);
e5_re = repmat([-1; zeros(n-1, 1)], n, 1);

idx_a = [1 n-1 n (N-n)];
idx_a_re = [(N-n) n n-1 1];
diag_a = [e2 e3 e1 e1];
diag_a_re = [e1 e1 e3_re e2_re];
A = (1/dxdy^2) * spdiags([diag_a_re -4*e1 diag_a], [-(idx_a_re) 0 idx_a], N, N);
A(1,1) = 2 * ((2*L)/ N);

idx_b = [n (N-n)];
idx_b_re = [(N-n) n];
diag_b = [e1 -e1];
diag_b_re = [-e1 e1];
B = (1/(2*dxdy)) * spdiags([-1*diag_b_re, diag_b], [-(idx_b_re), idx_b], N, N);

idx_c = [1, n-1];
idx_c_re = [n-1, 1];
diag_c = [e4 e5];
diag_c_re = [e5_re e2_re];
C = (1/(2*dxdy)) * spdiags([-1*diag_c_re, diag_c], [-(idx_c_re), idx_c], N, N);

x2 = linspace(-L/2, L/2, n+1); x = x2(1:n);
y=x;
[X, Y] = meshgrid(x, y);

% fourier space
kx = (2*pi/L)*[0:(n/2-1) (-n/2):-1]; 
kx(1)=10^-6;
ky=kx;
[KX, KY] = meshgrid(kx, ky);


rhsFun = @(psi,w)(-B*psi).*(C*w) + (C*psi).*(B*w) + v.*A*w;


%% Initial Conditions
a = 0.5;
b = -10;
c = -2.8;

w0 = exp(-0.5*(X - a*b).^2 - 0.1*(Y + c + b*b).^2) - exp(-0.5*(X - a*a).^2 - 0.1*(Y + c + b*a).^2);
w0 = w0 + rot90(-0.1*w0);
w0_vec = reshape(w0, N, 1);

[t,wCool_vec] = ode45(@(t,w) rhs5(t,w,rhsFun,KX,KY,n,N),tspan,w0_vec);

loops = size(wCool_vec,1);
F(loops) = struct('cdata',[],'colormap',[]);
for i = 1:loops
    w_curr = reshape(wCool_vec(i, :), n, n);
    set(gcf, 'Position',  [50, 100, 1150, 501]);
    ax1 = subplot(1, 2, 1);
    pcolor(X, Y, w_curr); 
    shading interp; colormap(lines(60)); axis off;
    
    ax2 = subplot(1, 2, 2);
    surf(X, Y, w_curr); 
    shading interp; colormap(lines(60)); axis off;
    zlim([-1, 1])
    
    pause(0.005);
    
    drawnow;
    F(i) = getframe(gcf);
end
video = VideoWriter('AsCoolAsItGets.avi', 'Uncompressed AVI');
video.FrameRate = 60;
open(video)
writeVideo(video, F);
close(video);


function f = rhs5(t,w,rhsFun,KX,KY,n,N)

    psi_f = -fft2(reshape(w,n,n))./(KX.^2+KY.^2);
    psi = real(ifft2(psi_f));
    psi = reshape(psi, N, 1);
   
    f = rhsFun(psi, w);
end