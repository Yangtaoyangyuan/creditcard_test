%--------V为原始训练集，V1和S2为已经分类的训练集，其中V1为label为0的类的训练集，
%S2为label为1的类训练集，对S2上采样200倍（395*200=79000），SM=[V1;S2]
%%随机分k折 将SM随机分为k份 
 clc
clearvars -except data Id indices S2 SM testV V V1 k %保留部分变量 
m=length(SM);
k=30;
indices=crossvalind('Kfold',m,k);%随机分K折
data=cell(1,k);%分k个单元
for i=1:k
    data{1,i}=[];
end
for i=1:m  %遍历indices，分配数据
    data{1,indices(i)}=[data{1,indices(i)};SM(i,:)];
end
%% k个子集中找出0-1变化位置
a=0;b=0;
count=zeros(k,1);
%把每折数据中的两类分开 data里的每折数据是label先为0后为1的排列顺序，因此找出由0变1的位置即可（count）
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
%% 朴素贝叶斯&交叉验证

 for z=1:30
    clearvars -except data Id indices S2 SM testV V V1 count z aaa k
    %将训练数据与测试数据分开 29份做训练 1份做测试
    traindata_0=zeros(1,29);
    traindata_1=zeros(1,29);
 for j=1:k
      n=length(data{1,j});
   if(j~=z)
    traindata_0=[traindata_0;data{1,j}(1:count(j,1),1:29)];%j!=z时为训练集
    traindata_1=[traindata_1;data{1,j}((count(j,1)+1):n,1:29)];
   else
    testdata_0=[data{1,z}(1:count(j,1),1:29)];%j=z时为训练集
    testdata_1=[data{1,z}(count(j,1)+1:n,1:29)];
   end
 end
 traindata_0(1,:)=[];%清除第一行
 traindata_1(1,:)=[];
 % ---------------------朴素贝叶斯-------------------------
 %分别计算两个类别训练集28个特征的均值和方差  假设28个特征均服从正态分布且互相独立
 for i=1:28
 [mu1(i,1),sigma1(i,1)]=normfit(traindata_0(:,i));
 [mu2(i,1),sigma2(i,1)]=normfit(traindata_1(:,i));
 end
 p1=length(traindata_0)/(length(traindata_0)+length(traindata_1));%求先验概率
 %计算后验概率 post为后验概率 prodt为似然函数
testdata=[testdata_0;testdata_1];
 m=length(data{1,z});
 answer=zeros(m,1);
 for i=1:m
 for j=1:28   
  prodt1(i,j)=normpdf(testdata(i,j),mu1(j,1),sigma1(j,1));%post1代表0类，post2代表1类
  post1(i,1)=p1*prod(prodt1(i,:));
  prodt2(i,j)=normpdf(testdata(i,j),mu2(j,1),sigma2(j,1));
  post2(i,1)=(1-p1)*prod(prodt2(i,:));
 end
 %交易金额amount条件概率计算
 c(i,1)=(length(find(traindata_0(:,29)==data{1,z}(i,29))))/length(traindata_0);
 d(i,1)=(length(find(traindata_1(:,29)==data{1,z}(i,29))))/length(traindata_1);
 if (c(i,)~=0)%若该交易金额在训练集中未曾出现则跳过
 post1(i,1)=post1(i,1)*c(i,1);
 post2(i,1)=post2(i,1)*d(i,1);
 end
 if (post1(i,1)>=post2(i,1)) %归一化因子相同，比较似然函数大小
     answer(i,1)=0;
 else
     answer(i,1)=1;
 end
 end
 %F1值计算
 true_answer=[zeros(count(z),1);ones(m-count(z),1)];
 TP=sum(answer(count(z)+1:m,1).*true_answer(count(z)+1:m,1));
 TFN=sum(xor(true_answer,answer));
 F1(z,1)=2*TP/(2*TP+TFN);
 end
 [F1max,z]=max(F1);%取F1最大时的z对应的训练集为测试数据的训练模型
 
%% ----------------用最佳训练模型测试数据集------------------------------
m=length(testV);
% testV_slash=testV(:,[4 5 6 7 9 11 12 14 16 18]);
testV_slash=testV;%28个特征可以筛除一些特征
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
  prodt1(i,j)=normpdf(testV_slash(i,j),mu1(j,1),sigma1(j,1));%post1代表0类，post2代表1类
  prodt2(i,j)=normpdf(testV_slash(i,j),mu2(j,1),sigma2(j,1));
 end
 post1(i,1)=p1*prod(prodt1(i,:));
  post2(i,1)=(1-p1)*prod(prodt2(i,:));
 %交易金额amount条件概率计算
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
TPN=sum(answer(:,1));%计算预测为1的总个数 


%% 导出csv文件
columns={'Id','Class'};
data=table(Id,answer,'VariableNames', columns);
writetable(data, 'submission.csv')
