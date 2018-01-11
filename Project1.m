clear; clc;
img=imread('puppy.png');
imshow(img)
title('Original')
img_gray=rgb2gray(img);
figure
imshow(img_gray)
title('Grayscale')
x_min=655; x_max=1014;
y_min=795; y_max=1154;
subim = img_gray(x_min:x_max,y_min:y_max,:);
[M,N]=size(img_gray);
figure
imshow(subim)
title('Subset of image')

%% 2
P=2; % Patch size
R=1; %Rate
C=2^(R*P^2);
patch_1=reshape(img_gray,[2,2,M*N/P^2]);
patch=reshape(patch_1,[1,4,M*N/P^2]);
[idx,rep]=kmeans(patch(:),C);
for i=1: length(patch)
    patch(1,:,i)=rep(idx(4*i));
end
patchy=reshape(patch,[2048,2048]);
figure
imshow(patchy)
title('Compressed image')
x_min=655; x_max=1014;
y_min=795; y_max=1154;
subimp = patchy(x_min:x_max,y_min:y_max,:);
figure
imshow(subimp)
title('New Subset')
D=immse(img_gray,patchy)/(M*N);

%% 3
b=0;
D=[0,0,0,0];
for i=1:4
    R=0.25; P=2; k=(R*P^2);
    C=2^k;
    patch_1=reshape(img_gray,[2,2,M*N/P^2]);
    patch=reshape(patch_1,[1,4,M*N/P^2]);
    [idx,rep]=kmeans(patch(:),C);
    for j=1:length(patch)
        patch(1,:,j)=rep(idx(4*j));
    end
    patchy=reshape(patch,[2048,2048]);
    figure;
    imshow(patchy);
    title(['Compressed image R=',num2str(R)]);
    x_min=655; x_max=1014;
    y_min=795; y_max=1154;
    subimp = patchy(x_min:x_max,y_min:y_max,:);
    figure;
    imshow(subimp);
    title(['New Subset R=',num2str(R)]);
   D(i)=immse(img_gray,patchy)/(M*N);
end
figure;
plot([0.25:1/4:1],D)
title('Coding Rate (R) vs Distortion (D)')
xlabel('R')
ylabel('Distortion')
%%

for i=1:7
    R=1; P=4; k=(R*P^2);
    C=2^k;
    patch_1=reshape(img_gray,[4,4,M*N/P^2]);
    patch=reshape(patch_1,[1,16,M*N/P^2]);
    [idx,rep]=kmeans(patch(:),C);
    for j=1:length(patch)
        patch(1,:,j)=rep(idx(16*j));
    end
    patchy=reshape(patch,[2048,2048]);
    figure;
    imshow(patchy);
    title(['Compressed image R=',num2str(R)]);
    x_min=655; x_max=1014;
    y_min=795; y_max=1154;
    subimp = patchy(x_min:x_max,y_min:y_max,:);
    figure;
    imshow(subimp);
    title(['New Subset R=',num2str(R)]);
    D(i)=immse(img_gray,patchy)/(M*N);
end
figure;
plot([1/16:1/16:7/16],D)
title('Coding Rate (R) vs Distortion (D)')
xlabel('R')
ylabel('Distortion')

%%
H=0;
P=2;
for i=1:C
    Ni=sum(idx(:)==i);
    pi=Ni/((M*N)/(P^2));
    H=H+pi*log2(pi);
end
H=H*-1;
R=H/P^2;
