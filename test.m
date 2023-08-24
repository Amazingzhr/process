% test.m
path = 'D:\testdata\chuanbo\817\';
%path = 'D:\testdata\230309\';
fileName1 = '2.dat';
file1 = fopen([path, fileName1],'rb');
rawSignal1 = fread(file1,Inf,'float'); 
q1 = rawSignal1(1:2:end);
q2 = rawSignal1(2:2:end);
q = complex(q1,q2);
sig = q(10000:30000);
y = stft(sig);
mesh(y)



rf_freq = 868.1e6;    % carrier frequency 470 MHz, used to correct clock drift
sf = 7;             % spreading factor SF7
bw = 125e3;         % bandwidth 125 kHz
fs = 250e3;           % sampling rate 1 MHz

phy = LoRaPHY(rf_freq, sf, bw, fs);
phy.has_header = 1;         % explicit header mode
phy.cr = 4;                 % code rate = 4/8 (1:4/5 2:4/6 3:4/7 4:4/8)
phy.crc = 1;                % enable payload CRC checksum
phy.preamble_len = 8;       % preamble: 8 basic upchirps

% % Encode payload [1 2 3 4 5]
% symbols = phy.encode((1:5)');
% fprintf("[encode] symbols:\n");
% disp(symbols);
% 
% % Baseband Modulation
% sig = phy.modulate(symbols);

% Demodulation
[symbols_d, cfo, netid] = phy.demodulate(sig);
fprintf("[demodulate] symbols:\n");
disp(symbols_d);

% Decoding
[data, checksum] = phy.decode(symbols_d);
fprintf("[decode] data:\n");
disp(data);
fprintf("[decode] checksum:\n");
disp(checksum);



feather(sig)