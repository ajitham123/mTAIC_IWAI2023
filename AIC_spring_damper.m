%%  mTAIC active inference controller for a spring mass damper system.

% Author: Ajith Anil Meera, Donders Institute for Brain, Cognition and
% Behaviour, Nijmegen

% Date: 31st August 2023
% Code accompanying the paper "Towards Metacognitive Robot Decision Making
% for Tool Selection" at IWAI 2023.


%%%%%%%%%%%%%
%%
clear all

task = 1; % 1 for position control, 2 for velocity control, 3 for acceleration control
tool = 1; % tools 1, 2 or 3
goal_x = [.5; 0];  % goal position and velocity 
Pi_g = diag([.01 .01]);  % goal precision (gain): tune it with dt for stability of Euler discretization
learn = 1;   % learning rate
dt = .01;    % sampling time
nt = 2000;   % number of time steps



% spring mass damper system
% k = .3; m = .25; b = .3;

if tool ==1
    k = .2; m = .4; b = .4; % tool 1
elseif tool == 2
    k = .3; m = .3; b = .2; % tool 2
elseif tool == 3
    k = .3; m = .2; b = .6; % tool 3
end

model.A = [0 1; -k/m -b/m];
model.B = [0; 1/m];
model.C = [1 0];




% discretize the system
sys_d = c2d(ss(model.A,model.B,model.C,[]),dt,'zoh');

nx = size(model.A,1);
nu = size(model.B,2);
ny = size(model.C,1);
model.u = ones(nu,nt);%repmat(1*sin(.1*(0:dt:(nt-1)*dt)),nu,1);  % k1 sin(omega*t)
model.x = zeros(nx,nt);
model.y = zeros(ny,nt);

a = zeros(size(model.u));
Pa = .00001*eye(nu);  % prior precision on action
% A = eye(nx);
Pi_w = 10*eye(nx);
k_h = 4;

Ug = 0; Uc = 0; H = 0; F_action = 0; 
for i = 3:nt-1
   
    % assuming a known state
    brain.x = model.x(:,i); % brain.x = Dx - dFdx
%     brain.x = - learn* (brain.x - goal_x)*Pi_g*dt;
    
    %%%% action from FEP (without priors on action) :dadt = -dFda = -dFdx*dxda
    
    
    if task == 1
        % active inference for constant goal state
%         a(:,i) = a(:,i-1) - learn*( (brain.x - goal_x)'*Pi_g*(-inv(model.A)*model.B) + Pa*a(:,i-1)  )*dt;
        k_h = 1;
        dFda = (brain.x - goal_x).'*Pi_g*(-pinv(model.A)*model.B) + Pa*a(:,i-1);
        dFdaa = (-pinv(model.A)*model.B).'*Pi_g*(-pinv(model.A)*model.B) + Pa;
        a(:,i) = a(:,i-1) + (expm(-k_h*dFdaa*dt)-eye(ny))*pinv(dFdaa)*dFda;  % action update from Free energy curve
            
        Pi_u = dFdaa; % precision on action
        Ug = Ug + .5*(model.x(:,i) - goal_x)'*Pi_g*(model.x(:,i) - goal_x); % performance
        Uc = Uc + .5*a(:,i)'*Pa*a(:,i); % control cost
        H = H -.5*log(det(Pi_u));       % confidence
        
    elseif task == 2
        % active inference for constant goal state velocity
        goal_vel = [10; 0];  Pi_vel = diag([1 1]);
        dFda = (model.A*ones(nx,nu)+model.B)'*Pi_vel*(model.A*brain.x + ...
                        model.B*a(:,i-1) - goal_vel) + Pa*a(:,i-1);
        dFdaa = (model.A*ones(nx,nu)+model.B)'*Pi_vel*(model.A*ones(nx,nu)+model.B) + Pa;
%         dFda = (model.A*(-pinv(model.A)*model.B)+model.B)'*Pi_vel*(model.A*brain.x + ...
%                         model.B*a(:,i-1) - goal_vel) + Pa*a(:,i-1);
%         dFdaa = (model.A*(-pinv(model.A)*model.B)+model.B)'*Pi_vel*(model.A*(-pinv(model.A)*model.B)+model.B) + Pa;

        a(:,i) = a(:,i-1) + (expm(-k_h*dFdaa*dt)-eye(nu))*pinv(dFdaa)*dFda;
        Pi_u = dFdaa;
        Ug = Ug +  .5*([model.x(2,i); (model.x(2,i)-model.x(2,i-1))/dt] - goal_vel)'...
               *Pi_vel*([model.x(2,i); (model.x(2,i)-model.x(2,i-1))/dt] - goal_vel);
        Uc = Uc + .5*a(:,i)'*Pa*a(:,i);
        H = H -.5*log(det(Pi_u));
        
    elseif task == 3
        % active inference for constant goal state acceleration
        goal_acc = [.5; 0];  Pi_acc = diag([10 0]);
%         Pa = .1; k_h = 4;
        a_dot = (a(:,i-1) - a(:,i-2))/dt;
        dFda = (model.A^2*[1; 0] +model.A*model.B + model.B*0  )'*Pi_acc*...
            (model.A*(model.A*brain.x + model.B*a(:,i-1)) + model.B*a_dot - goal_acc)  + Pa*a(:,i-1);
        dFdaa = (model.A^2*[1; 0] +model.A*model.B + model.B*0 )'*Pi_acc*....
            (model.A^2*[1; 0] +model.A*model.B + model.B*0 )  + Pa;
        a(:,i)= a(:,i-1) + (expm(-k_h*dFdaa*dt)-eye(ny))*pinv(dFdaa)*dFda;
        
        Pi_u = dFdaa;
        Ug = Ug + .5*([(model.x(2,i)-model.x(2,i-1))/dt; 0] - goal_acc)'*Pi_acc*...
                ([(model.x(2,i)-model.x(2,i-1))/dt; 0] - goal_acc);
        Uc = Uc + .5*a(:,i)'*Pa*a(:,i);
        H = H -.5*log(det(Pi_u));
    end
    %     A = A - learn * (brain.x - goal_x)'*Pi_g* ( -inv(A) * brain.x )*dt
    %     A = A + learn * (brain.x'*Pi_w*([brain.x(2,1); 0] -A*brain.x-model.B*a(:,i-1)) )*dt   ;
    
    % Generative process - take action in the world
    model.x(:,i+1) = sys_d.A*model.x(:,i) + sys_d.B*a(:,i);
%     model.x(:,i+1) = sys_d.A*model.x(:,i) + sys_d.B*model.u(:,i);
    model.y(:,i)   = sys_d.C*model.x(:,i);% + .001*(-1+2*rand(ny,1));  
    

    
end
a(:,i+1) = a(:,i);
F_action = F_action + Ug + Uc + H; % time integral of free energy

%%
% Plot the results
figure(2);  subplot(1,4,1); 
plot(0:dt:(nt-1)*dt,model.x(1,:),'linewidth',2); hold on; 
xlabel('time (s)'); ylabel('x')
if task == 1
    yline(goal_x(1),'--','linewidth',2); hold on; 
    subplot(1,4,2); yline(goal_x(2),'--','linewidth',2); hold on;
end

subplot(1,4,2)
plot(0:dt:(nt-1)*dt,model.x(2,:)','linewidth',2); hold on; 
xlabel('time (s)'); ylabel('$\dot{x}$','interpreter','latex')
if task == 2
    yline(goal_vel(1),'--','linewidth',2);
    subplot(1,4,3); yline(goal_x(2),'--','linewidth',2); hold on;
end

subplot(1,4,3)
plot(dt:dt:(nt-1)*dt,diff(model.x(2,:))'/dt,'linewidth',2); 
hold on; xlabel('time (s)'); ylabel('$\ddot{x}$','interpreter','latex')
if task == 3
    yline(goal_acc(1),'--','linewidth',2);
end

subplot(1,4,4);
% fill([0:dt:(nt-1)*dt, fliplr(0:dt:(nt-1)*dt)], [a+Pi_u^-.5, fliplr(a-Pi_u^-.5)],...
%     [.5 .5 .5],'FaceAlpha',0.1,'linewidth',.01);
% hold on;
plot(0:dt:(nt-1)*dt,a,'linewidth',2); xlabel('time (s)'); ylabel('u '); hold on;
set(findobj(gcf,'type','axes'),'FontSize',12);

fprintf('\n U^g, U^c,H,F,Pi^u = [%f,%f,%f,%f,%f]\n',Ug,Uc,H,F_action,Pi_u)
