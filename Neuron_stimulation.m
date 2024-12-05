%% Neuron Action Potential Simulation
clc;
clear;
close all;

%% euler's method
ini_time = -100;
fin_time = 100;
time_step = 0.01;
time_span = ini_time:time_step:fin_time;

der_f_t = fun_der_f_t(time_span);

tiledlayout(2,1);
nexttile;
plot(time_span, der_f_t, 'Color', 'b');
grid on;
xlabel('x');
title('f\prime(x)');


han_der_f_t = @fun_der_f_t;
ini_f_t = (ini_time^3)/3;
ode_solve_der_f_t = ode_euler(han_der_f_t, time_span, time_step, ini_f_t);

nexttile
hold on;
plot(time_span, ode_solve_der_f_t,  'LineWidth', 6, 'Color', 'r');
plot(time_span, (time_span.^3)/3, 'LineWidth', 3, 'Color', 'c');
grid on;
legend('ODE Solver', 'Analytical');
title('f(x)');
xlabel('x');
hold off;

%% creating handle
han_n_prime = @n_prime;

%% current response of K+ channel, constant voltage
ini_time = 0;
fin_time = 20;
time_step = 0.01;
ini_n = 0;
Ek = -72.1;
gk_bar = 36; 

v_hold = -30;
cur_res_K = K_v(han_n_prime, ini_time, fin_time, time_step, ini_n, v_hold, Ek, gk_bar);

figure;
plot(ini_time:time_step:fin_time, cur_res_K, 'LineWidth', 3);
title('Current Response of K+ channel, V_h_o_l_d = -30 mv');
xlabel('Time (ms)');
ylabel('Current (mA)');
grid on;


%% current response of K+ channel, multiple constant voltage
figure;
hold on;
legend_str = {};
for v_hold = -30:10:70
    cur_res_K = K_v(han_n_prime, ini_time, fin_time, time_step, ini_n, v_hold, Ek, gk_bar);

    plot(ini_time:time_step:fin_time, cur_res_K, 'LineWidth', 3, 'Color', rand(1,3));
    legend_str = cat(2, legend_str, cat(2, 'V_h = ', num2str(v_hold), ' mV'));
end
legend(legend_str);
hold off;
title('Current Response of K+ channel');
xlabel('Time (ms)');
ylabel('Current (mA)');
grid on;

%% creating handle
han_m_prime = @m_prime;
han_h_prime = @h_prime;

%% current response of Na+ channel, constant voltage
ini_time = 0;
fin_time = 20;
time_step = 0.01;
ini_m = 0;
ini_h = 1;
Ena = 52.4;
gna_bar = 120;

v_hold = -30;
cur_res_Na = ...
    Na_v(han_m_prime, han_h_prime, ini_time, fin_time, time_step, ini_m, ini_h, v_hold, Ena, gna_bar);

figure;
plot(ini_time:time_step:fin_time, cur_res_Na, 'LineWidth', 3);
title(cat(2,'Current Response of Na+ channel, V_h_o_l_d = ', num2str(v_hold), ' mV'));
xlabel('Time (ms)');
ylabel('Current (mA)');
grid on;

%% current response of Na+ channel, multiple constant voltage
figure;
hold on;
legend_str = {};
for v_hold = -30:10:70
    cur_res_Na = ...
        Na_v(han_m_prime, han_h_prime, ini_time, fin_time, time_step, ini_m, ini_h, v_hold, Ena, gna_bar);

    plot(ini_time:time_step:fin_time, cur_res_Na, 'LineWidth', 3, 'Color', rand(1,3));
    legend_str = cat(2, legend_str, cat(2, 'V_h = ', num2str(v_hold), ' mV'));
end
legend(legend_str);
hold off;
title('Current Response of Na+ channel');
xlabel('Time (ms)');
ylabel('Current (mA)');
grid on;

%% peaks of late (K+) and early (Na+) currents
ini_time = 0;
fin_time = 20;
time_step = 0.01;

ini_n = 0;
ini_m = 0;
ini_h = 1;

Ek = -72.1;
gk_bar = 36;
Ena = 52.4;
gna_bar = 120;
v_hold = -100:1:70;

I_max_K = zeros(1, size(v_hold, 2));
I_min_K = zeros(1, size(v_hold, 2));
I_max_Na = zeros(1, size(v_hold, 2));
I_min_Na = zeros(1, size(v_hold, 2));
for i = 1:size(v_hold, 2)
    cur_res_K = K_v(han_n_prime, ini_time, fin_time, time_step, ini_n, v_hold(i), Ek, gk_bar);
    cur_res_Na = ...
        Na_v(han_m_prime, han_h_prime, ini_time, fin_time, time_step, ini_m, ini_h, v_hold(i), Ena, gna_bar);
    
    I_max_K(i) = max(cur_res_K, [], "all");
    I_min_K(i) = min(cur_res_K, [], "all");
    I_max_Na(i) = max(cur_res_Na, [], "all");
    I_min_Na(i) = min(cur_res_Na, [], "all");
end

figure;
hold on;
plot(v_hold, I_max_K, 'LineWidth', 3);
plot(v_hold, I_min_K, 'LineWidth', 6);
plot(v_hold, I_max_Na, 'LineWidth', 3);
plot(v_hold, I_min_Na, 'LineWidth', 3);
legend('I_m_a_x K+', 'I_m_i_n K+', 'I_m_a_x Na+', 'I_m_i_n Na+');
hold off;
title('Peaks of Late (K+) and Early (Na+) Currents');
xlabel('Voltage (mV)');
ylabel('Current (mA)');
grid on;

%% better implementation, peaks of late (K+) and early (Na+) currents
ini_time = 0;
fin_time = 20;
time_step = 0.01;

ini_n = 0;
ini_m = 0;
ini_h = 1;

Ek = -72.1;
gk_bar = 36;
Ena = 52.4;
gna_bar = 120;
v_hold = -100:1:70;

I_max_K = zeros(1, size(v_hold, 2));
I_max_Na = zeros(1, size(v_hold, 2));
for i = 1:size(v_hold, 2)
    cur_res_K = K_v(han_n_prime, ini_time, fin_time, time_step, ini_n, v_hold(i), Ek, gk_bar);
    cur_res_Na = ...
        Na_v(han_m_prime, han_h_prime, ini_time, fin_time, time_step, ini_m, ini_h, v_hold(i), Ena, gna_bar);
    
    [I_max_K(i), index] = max(abs(cur_res_K), [], "all");
    I_max_K(i) = I_max_K(i) * sign(cur_res_K(index));
    I_max_Na(i) = max(abs(cur_res_Na), [], "all");
    I_max_Na(i) = I_max_Na(i) * sign(cur_res_Na(index));
end

figure;
hold on;
plot(v_hold, I_max_K, 'LineWidth', 3);
plot(v_hold, I_max_Na, 'LineWidth', 3);
legend('I_p_e_a_k K+', 'I_p_e_a_k Na+');
hold off;
title('Peaks of Late (K+) and Early (Na+) Currents');
xlabel('Voltage (mV)');
ylabel('Current (mA)');
grid on;

%% hodgkin huxley voltage, constant injection current, Vhold = -60
ini_time = 0;
fin_time = 100;
time_step = 0.01;

ini_n = 0;
ini_m = 0;
ini_h = 1;

Cm = 1;
Ek = -72.1;
gk_bar = 36;
Ena = 52.4;
gna_bar = 120;
gl_bar = 0.3;
El = -49.2;

ini_v = -60;
I_inj = 5;


v_hodgkin_huxley = hodgkin_huxley(han_n_prime, han_m_prime,  han_h_prime, ...
    ini_time, fin_time, time_step, ...
    ini_n, ini_m, ini_h, ini_v, ...
    Cm, Ek, gk_bar, Ena, gna_bar, gl_bar, El, I_inj);

figure;
plot(ini_time:time_step:fin_time, v_hodgkin_huxley, 'LineWidth', 3);
title(cat(2,'Hodgkin Huxley Voltage, I_i_n_j = ', num2str(I_inj), ' mA'));
xlabel('Time (ms)');
ylabel('Voltage (mV)');
grid on;

%% hodgkin huxley voltage, multiple constant injection current, Vhold = -60
ini_time = 0;
fin_time = 200;
time_step = 0.01;

ini_n = 0;
ini_m = 0;
ini_h = 1;

Cm = 1;
Ek = -72.1;
gk_bar = 36;
Ena = 52.4;
gna_bar = 120;
gl_bar = 0.3;
El = -49.2;

ini_v = -60;

figure;
hold on;
legend_str = {};
for I_inj = 5:5:15
    v_hodgkin_huxley = hodgkin_huxley(han_n_prime, han_m_prime,  han_h_prime, ...
        ini_time, fin_time, time_step, ...
        ini_n, ini_m, ini_h, ini_v, ...
        Cm, Ek, gk_bar, Ena, gna_bar, gl_bar, El, I_inj);

    plot(ini_time:time_step:fin_time, v_hodgkin_huxley, 'LineWidth', 3);
    legend_str = cat(2, legend_str, cat(2, 'I_i_n_j = ', num2str(I_inj), ' mA'));
end
legend(legend_str);
hold off;
title('Hodgkin Huxley Voltage');
xlabel('Time (ms)');
ylabel('Voltage (mV)');
grid on;

%% FitzHugh–Nagumo model
ini_time = 0;
fin_time = 200;
time_step = 0.01;

figure;
hold on;
legend_str = {};
for I_inj = [0 3  5 12 20]
    v_FitzHugh = FitzHugh(ini_time, fin_time, time_step, I_inj);

    plot(ini_time:time_step:fin_time, v_FitzHugh, 'LineWidth', 3);
    legend_str = cat(2, legend_str, cat(2, 'I_i_n_j = ', num2str(I_inj), ' mA'));
end
legend(legend_str);
hold off;
title('FitzHugh–Nagumo Voltage');
xlabel('Time (ms)');
ylabel('Voltage (mV)');
grid on;

%% bursting type voltage
ini_time = 1;
fin_time = 20;
time_step = 1e-5;

figure;
hold on;
legend_str = {};
for I_inj = [0 5 10]
    v_bursting = bursting(ini_time, fin_time, time_step, I_inj);

    plot(ini_time:time_step:fin_time, v_bursting, 'LineWidth', 1);
    legend_str = cat(2, legend_str, cat(2, 'I_i_n_j = ', num2str(I_inj), ' mA'));
end
legend(legend_str);
hold off;
title('Bursting Type Voltage');
xlabel('Time (ms)');
ylabel('Voltage (mV)');
ylim([-70 40]);
grid on;

%% alpha_n & beta_n of n-prime (in Derivative of K+/Late Current)
v_span = -100:0.1:40;
v_rest = -60;
epsilon = 1e-9;

vm = v_span - v_rest + epsilon;

alpha_n = 0.01 .* (10-vm) ./ ( exp((10-vm)/10)-1 );
beta_n = 0.125 .* exp(-vm/80);

figure;
hold on;
plot(v_span, alpha_n, 'LineWidth', 3);
plot(v_span, beta_n, 'LineWidth', 3);
legend({'\alpha_n', '\beta_n'});
title('\alpha_n & \beta_n of n\prime (in Derivative of K+/Late Current)');
ylabel('Probability');
xlabel('Voltage (mV)');
grid on;


%% function declaration
function v_bursting = bursting(ini_time, fin_time, time_step, I_inj)
    v_rest = -65;
    a = 0.02;
    b = 0.2;
    c = -55;
    d = 4;
    
    time_span = ini_time:time_step:fin_time;
        
    v_t = zeros(1, size(time_span, 2)+1);
    u_t = zeros(1, size(time_span, 2)+1);
    
    v_t(1) = v_rest;
    u_t(1) = b * v_rest;
    
    for i = 2:(size(time_span, 2)+1)
        der_v_t = (0.4*(v_t(i-1)^2)) + 5*v_t(i-1) + 140 - u_t(i-1) + I_inj;
        der_u_t = a * (b*v_t(i-1) - u_t(i-1));
    
        v_t(i) = v_t(i-1) + der_v_t * time_step;
        u_t(i) = u_t(i-1) + der_u_t * time_step;
    
        if v_t(i) >= 30
            v_t(i) = c;
            u_t(i) = u_t(i) + d;
        end
    end

    v_bursting = v_t(2:end);
end

function v_FitzHugh = FitzHugh(ini_time, fin_time, time_step, I_inj)
    a = 0.7;
    b = 0.8;
    R = 0.1;
    t = 12.5;
    
    time_span = ini_time:time_step:fin_time;
    
    v_t = zeros(1, size(time_span, 2)+1);
    w_t = zeros(1, size(time_span, 2)+1);
    
    v_t(1) = 0;
    w_t(1) = 0;
    
    for i = 2:(size(time_span, 2)+1)
        der_v_t = v_t(i-1) - ((v_t(i-1)^3)/3) - w_t(i-1) + (R * I_inj);
        der_w_t = (v_t(i-1) + a - (w_t(i-1) * b)) / t;
    
        v_t(i) = v_t(i-1) + der_v_t * time_step;
        w_t(i) = w_t(i-1) + der_w_t * time_step;
    end
    v_FitzHugh = v_t(2:end);
end

function v_hodgkin_huxley = hodgkin_huxley(n_prime, m_prime,  h_prime, ...
    ini_time, fin_time, time_step, ...
    ini_n, ini_m, ini_h, ini_v, ...
    Cm, Ek, gk_bar, Ena, gna_bar, gl_bar, El, I_inj)

    time_span = ini_time:time_step:fin_time;

    v_t = zeros(1, size(time_span, 2)+1);
    v_t(1) = ini_v;
    n_t = ini_n;
    m_t = ini_m;
    h_t = ini_h;
    
    for i = 2:size(time_span, 2)+1
        d_v_t = (-1/Cm) * (  gl_bar * (v_t(i-1) - El) + ...
            gk_bar * n_t^4 * (v_t(i-1) - Ek) + ...
            gna_bar * m_t^3 * h_t * (v_t(i-1) - Ena) + ...
            (-I_inj)  );
    
        v_t(i) = v_t(i-1) + d_v_t * time_step;
        n_t = n_t + n_prime(n_t, v_t(i-1)) * time_step;
        m_t = m_t + m_prime(m_t, v_t(i-1)) * time_step;
        h_t = h_t + h_prime(h_t, v_t(i-1)) * time_step;
    end
    
    v_hodgkin_huxley = v_t(2:end);
end

function cur_res_Na = Na_v(m_prime, h_prime, ini_time, fin_time, time_step, ini_m, ini_h, v_hold, Ena, gna_bar)
    time_span = ini_time:time_step:fin_time;

    cur_res_m_Na = ode_euler2(m_prime, time_span, time_step, ini_m, v_hold);
    cur_res_h_Na = ode_euler2(h_prime, time_span, time_step, ini_h, v_hold);

    cur_res_Na = 1e-3 * (v_hold - Ena) * gna_bar * (cur_res_m_Na.^3) .* cur_res_h_Na;
end

function h_prime_value = h_prime(h_v, v)
    v_rest = -60;
    epsilon = 1e-9;

    vm = v - v_rest + epsilon;

    alpha_h = 0.7 * ( exp(-vm/20) );
    beta_h = (exp( (30-vm)/10 ) + 1) ^ (-1);
    
    h_prime_value = (alpha_h*(1-h_v))-(beta_h*h_v);
end

function m_prime_value = m_prime(m_v, v)
    v_rest = -60;
    epsilon = 1e-9;

    vm = v - v_rest + epsilon;

    alpha_m = 0.1 * (25-vm) / ( exp(0.1*(25-vm))-1 );
    beta_m = 4 * exp(-vm/18);
    
    m_prime_value = (alpha_m*(1-m_v))-(beta_m*m_v);
end

function cur_res_K = K_v(n_prime, ini_time, fin_time, time_step, ini_n, v_hold, Ek, gk_bar)
    time_span = ini_time:time_step:fin_time;

    cur_res_n_K = ode_euler2(n_prime, time_span, time_step, ini_n, v_hold);

    cur_res_K = 1e-3 * (v_hold - Ek) * gk_bar * (cur_res_n_K.^4);
end

function ode_solve_der_f_t = ode_euler2(fun_der_f_t, time_span, time_step, ini_f_t, v_hold)
    ode_solve_der_f_t = zeros(1, size(time_span, 2));

    ode_solve_der_f_t(1) = ini_f_t + fun_der_f_t(ini_f_t, v_hold)*time_step;
    for i = 2:size(ode_solve_der_f_t, 2)
        ode_solve_der_f_t(i) = ode_solve_der_f_t(i-1) + fun_der_f_t(ode_solve_der_f_t(i-1),v_hold)*time_step;
    end
end

function n_prime_value = n_prime(n_v, v)
    v_rest = -60;
    epsilon = 1e-9;

    vm = v - v_rest + epsilon;

    alpha_n = 0.01 * (10-vm) / ( exp((10-vm)/10)-1 );
    beta_n = 0.125 * exp(-vm/80);
    
    n_prime_value = (alpha_n*(1-n_v))-(beta_n*n_v);
end

function ode_solve_der_f_t = ode_euler(fun_der_f_t, time_span, time_step, ini_f_t)
    der_f_t = fun_der_f_t(time_span);

    ode_solve_der_f_t = zeros(1, size(time_span, 2));
    ode_solve_der_f_t(1) = ini_f_t + der_f_t(1)*time_step;
    for i = 2:size(ode_solve_der_f_t, 2)
        ode_solve_der_f_t(i) = ode_solve_der_f_t(i-1) + der_f_t(i)*time_step;
    end
end

function der_f_t = fun_der_f_t(x_span)
    der_f_t = (x_span.^2);
end