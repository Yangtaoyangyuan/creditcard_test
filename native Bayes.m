%--------VΪԭʼѵ������V1��S2Ϊ�Ѿ������ѵ����������V1ΪlabelΪ0�����ѵ������
%S2ΪlabelΪ1����ѵ��������S2�ϲ���200����395*200=79000����SM=[V1;S2]
%%�����k�� ��SM�����Ϊk�� 
 clc
clearvars -except data Id indices S2 SM testV V V1 k %�������ֱ��� 
m=length(SM);
k=30;
indices=crossvalind('Kfold',m,k);%�����K��
data=cell(1,k);%��k����Ԫ
for i=1:k
    data{1,i}=[];
end
for i=1:m  %����indices����������
    data{1,indices(i)}=[data{1,indices(i)};SM(i,:)];
end
%% k���Ӽ����ҳ�0-1�仯λ��
a=0;b=0;
count=zeros(k,1);
%��ÿ�������е�����ֿ� data���ÿ��������label��Ϊ0��Ϊ1������˳������ҳ���0��1��λ�ü��ɣ�count��
for i=1:k
    n=length(data{1,i});
    for j=1:n
    if (data{1,i}(j,30)==0)
        count(i,1)=count(i,1)+1;
    else
        break
    end
    end
end
%% ���ر�Ҷ˹&������֤

 for z=1:30
    clearvars -except data Id indices S2 SM testV V V1 count z aaa k
    %��ѵ��������������ݷֿ� 29����ѵ�� 1��������
    traindata_0=zeros(1,29);
    traindata_1=zeros(1,29);
 for j=1:k
      n=length(data{1,j});
   if(j~=z)
    traindata_0=[traindata_0;data{1,j}(1:count(j,1),1:29)];%j!=zʱΪѵ����
    traindata_1=[traindata_1;data{1,j}((count(j,1)+1):n,1:29)];
   else
    testdata_0=[data{1,z}(1:count(j,1),1:29)];%j=zʱΪѵ����
    testdata_1=[data{1,z}(count(j,1)+1:n,1:29)];
   end
 end
 traindata_0(1,:)=[];%�����һ��
 traindata_1(1,:)=[];
 % ---------------------���ر�Ҷ˹-------------------------
 %�ֱ�����������ѵ����28�������ľ�ֵ�ͷ���  ����28��������������̬�ֲ��һ������
 for i=1:28
 [mu1(i,1),sigma1(i,1)]=normfit(traindata_0(:,i));
 [mu2(i,1),sigma2(i,1)]=normfit(traindata_1(:,i));
 end
 p1=length(traindata_0)/(length(traindata_0)+length(traindata_1));%���������
 %���������� postΪ������� prodtΪ��Ȼ����
testdata=[testdata_0;testdata_1];
 m=length(data{1,z});
 answer=zeros(m,1);
 for i=1:m
 for j=1:28   
  prodt1(i,j)=normpdf(testdata(i,j),mu1(j,1),sigma1(j,1));%post1����0�࣬post2����1��
  post1(i,1)=p1*prod(prodt1(i,:));
  prodt2(i,j)=normpdf(testdata(i,j),mu2(j,1),sigma2(j,1));
  post2(i,1)=(1-p1)*prod(prodt2(i,:));
 end
 %���׽��amount�������ʼ���
 c(i,1)=(length(find(traindata_0(:,29)==data{1,z}(i,29))))/length(traindata_0);
 d(i,1)=(length(find(traindata_1(:,29)==data{1,z}(i,29))))/length(traindata_1);
 if (c(i,)~=0)%���ý��׽����ѵ������δ������������
 post1(i,1)=post1(i,1)*c(i,1);
 post2(i,1)=post2(i,1)*d(i,1);
 end
 if (post1(i,1)>=post2(i,1)) %��һ��������ͬ���Ƚ���Ȼ������С
     answer(i,1)=0;
 else
     answer(i,1)=1;
 end
 end
 %F1ֵ����
 true_answer=[zeros(count(z),1);ones(m-count(z),1)];
 TP=sum(answer(count(z)+1:m,1).*true_answer(count(z)+1:m,1));
 TFN=sum(xor(true_answer,answer));
 F1(z,1)=2*TP/(2*TP+TFN);
 end
 [F1max,z]=max(F1);%ȡF1���ʱ��z��Ӧ��ѵ����Ϊ�������ݵ�ѵ��ģ��
 
%% ----------------�����ѵ��ģ�Ͳ������ݼ�------------------------------
m=length(testV);
% testV_slash=testV(:,[4 5 6 7 9 11 12 14 16 18]);
testV_slash=testV;%28����������ɸ��һЩ����
traindata_0_slash=traindata_0;
traindata_1_slash=traindata_1;
[p,q]=size(testV_slash);
 for i=1:q
 [mu1(i,1),sigma1(i,1)]=normfit(traindata_0_slash(:,i));
 [mu2(i,1),sigma2(i,1)]=normfit(traindata_1_slash(:,i));
 end
 p1=length(traindata_0)/(length(traindata_0)+length(traindata_1));
for i=1:m
 for j=1:q
  prodt1(i,j)=normpdf(testV_slash(i,j),mu1(j,1),sigma1(j,1));%post1����0�࣬post2����1��
  prodt2(i,j)=normpdf(testV_slash(i,j),mu2(j,1),sigma2(j,1));
 end
 post1(i,1)=p1*prod(prodt1(i,:));
  post2(i,1)=(1-p1)*prod(prodt2(i,:));
 %���׽��amount�������ʼ���
 c(i,1)=(length(find(traindata_0(:,29)==testV(i,29))))/length(traindata_0);
 d(i,1)=(length(find(traindata_1(:,29)==testV(i,29))))/length(traindata_1);
 post1(i,1)=post1(i,1)*c(i,1);
 post2(i,1)=post2(i,1)*d(i,1);
 if (post1(i,1)>=post2(i,1))
     answer(i,1)=0;
 else
     answer(i,1)=1;
 end
end
TPN=sum(answer(:,1));%����Ԥ��Ϊ1���ܸ��� 


%% ����csv�ļ�
columns={'Id','Class'};
data=table(Id,answer,'VariableNames', columns);
writetable(data, 'submission.csv')
