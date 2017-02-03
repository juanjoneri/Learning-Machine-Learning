%% Machine Learning Online Class - Exercise 1: Linear Regression

%% Initialization
clear ; close all; clc

%% ======================= Part 2: Plotting =======================
data = load('ex1data2.txt');
th0 = data(:, 1:2); 
Y = data(:, 3);
m = length(Y); % number of training examples
X = [ones(m,1),th0(:,1),th0(:,2)];
%Ploting area vs price
figure
plot(X(:,2), Y, 'rx', 'MarkerSize', 10); % Plot the data 

%Ploting rooms vs price
figure
plot(X(:,3), Y, 'ro', 'MarkerSize', 10); % Plot the data 


%% =================== Part 1: Normalizing and Calling cost function ===================
 X = normalizeFeatures(X)
n=100;
m=30;
th1=linspace(570000,610000,n);
th2=linspace(-140000,-80000,n);
th0 = 350000;
z=zeros(1,n);
for i=1:n
    for j=1:n
        z(j,i)=CostFunction([th0,th1(i),th2(j)],X,Y);
    end
end

figure
surf(th1,th2,z)
xlabel('x')
ylabel('y')
zlabel('z')

figure
C=contourf(z)
clabel(C)

figure
contour3(z)
