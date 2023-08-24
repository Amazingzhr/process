path = 'D:\testdata\chuanbo\84\';
fileName1 = '100.dat';
file1 = fopen([path, fileName1],'rb');
rawSignal1 = fread(file1,Inf,'float'); 
q1 = rawSignal1(1:2:end);
q2 = rawSignal1(2:2:end);
q = complex(q1,q2);%读取数据

% feather(q(16000:72000))%选部分数据画出原始图像
%feather(q(1:30000))
% figure(20)
% plot(abs(q(12000:25000)))

% q10 = q(16000:20000);q11 = q(42000:46000);q12 = q(67000:70000);
% qtest = [q10; q11; q12];
% feather(qtest)




%% 短时傅里叶变换
window = 1000;%窗函数长度
overlap = 800;%重叠长度
nfft = window;%DFT点数
fs = 250000;%采样率
[S1,F1,T1,P1] = spectrogram(q(5:200000) ,window,overlap,nfft ,fs );

figure(4)
surf(T1,F1,20*log10((abs(S1))),'edgecolor','none','FaceColor','interp')
xlabel('时间t/s')
ylabel('频率f/Hz')
zlabel('强度P/dBm')
title('时频图')
view(2)
colormap('jet')


H = qtest;
%P = abs((H).^2)/length(H)^2;
P = abs((H).^2);
Pf = sum(P,2)/length(H);

t = (1:length(H))/250000;

plot(t,Pf)
xlabel('时间t/s')
ylabel('幅值')
title('幅值-时间图')


%% 生成down_chirp
% d_sf = 7; % spreading factor
% d_fft_size = 2^d_sf; % fft size
% accumulator = 0;
% phase = -pi;
% sig_downchirp = zeros(2*d_fft_size,1);
% for ii= 1:2*d_fft_size
%     accumulator = accumulator + phase;
%     sig_downchirp(ii) = exp(-1i*accumulator);
%     phase = phase + 2*pi/d_fft_size;
% end
% feather(sig_downchirp)
% 
% C = conv(sig_downchirp,rawSignal1,'full');
% 
% H = abs(C);

%% 求速度时间图
A = zeros();
G = [];
for    i =  1000:length(H)
    if mod(i,1000) == 0
        [C000,~] = autocorr(H(i-999:i),999);%信号的自相关函数
       % D000 = diff(C000);
            X = smooth(C000,30,'sgolay',3);
            Y = smooth(X,10,'sgolay',2);
            Z = movmean(Y,5);
           D000 = diff(Z);%求差分
        G=[G Z];
           [N,Loc] = findpeaks(D000,'MinPeakDistance',0.001);%找波峰
           % plot(D000)
           % [M,LOC] = findpeaks(-D000);

           
         
         if length(Loc) < 2   && N(1) < 0  %判别
%              isempty(Loc)
              A(end+1) = 0;
         elseif length(Loc) < 2   && N(1) > 0
              A(end+1) = 14/Loc(1);
        
         elseif length(Loc) > 1   && N(2) < 0.01
             %Z(Loc(5))-Z(Loc(4)) < 0.0101
            % m = Loc(1);n = LOC(1);q = Z(m)-Z(n);q < 0.05;             
             A(end+1) = 0;
         else
                    %plot(C000)
%         xlabel('采样点数（2秒）')
%         ylabel('自相关值')
        A(end+1) = 14/Loc(2);      %求速度
        end
    end
end
%画图
t = (1:length(A))/5;
B = hampel(A,10,2);
C = hampel(B,3,2);
figure(1)
plot(t,C,'LineWidth',2)
hold on 
scatter(t,A,'r')
%xlim([0 15])
%ylim([0 1.2])
xlabel('时间t/s')
ylabel('速度m/s')
title('速度-时间图')



%% 拿出一部分数据看自相关效果
figure(2)
 [C0,~] = autocorr(qtest(1000:5000),3999);%信号的自相关函数

   X1 = smooth(C0,30,'sgolay',3);
   Y1 = smooth(X1,10,'sgolay',2);
   Z1 = movmean(Y1,5);
Z2 = diff(Z1);
plot(Z1)

[N0,Loc0] = findpeaks(Z2,'MinPeakDistance',0.001);
xlabel('延迟t/s')
ylabel('自相关值')
title('自相关值随延迟变化图')



%%
% %再做STFT
% window = 16;%窗函数长度
% overlap = 10;%重叠长度
% nfft = window;%DFT点数
% fs = 128;%采样率
% [S1,F1,T1,P1] = spectrogram(sig_downchirp ,window,overlap,nfft ,fs );
% 
% figure(4)
% surf(T1,F1,20*log10((abs(S1))),'edgecolor','none','FaceColor','interp')
% xlabel('时间t/s')
% ylabel('频率f/Hz')
% zlabel('强度P/dBm')
% title('时频图')
% %view(2)
% colormap('jet')


 %%
% d_sf = 7; % spreading factor
% d_fft_size = 2^d_sf; % fft size
% accumulator = 0;
% phase = -pi;
% sig_downchirp = zeros(2*d_fft_size,1);
% for ii= 1:2*d_fft_size
%     accumulator = accumulator + phase;
%     sig_downchirp(ii) = exp(-1i*accumulator);
%     phase = phase + 2*pi/d_fft_size;
% end


% C = conv(sig_downchirp,rawSignal1,'full');
% figure(5)
% t2 = (1:length(C))/500000;
% plot(t2,abs(C))
% xlabel('时间t/s')
% ylabel('幅值')
% title('幅值-时间图')

