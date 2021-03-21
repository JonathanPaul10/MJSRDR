function [U1,U2] = MJSRDR(train_all,para_list,W)
%% info:
% ---input---
% train_all : it's the set of training samples
%               for a set of two-dimensional images, 
%               it should be palced in the form of train_all(:,:,i)(i=1:N)
% para_list : a vector with 5 elements: 
%                    parameter 1,2;
%                    dimension of features extracted;
%                    training number
%            参数1，2；降维数1，2，训练样本数
% W : 近邻信息矩阵
% ---output---
% U1: 一维方向 The projection matrix of the first mode
% U2: 二维方向 The projection matrix of the second mode
%% function content
global pen;
global h;
global w;
%para_list
para(1) = para_list(1);
para(2) = para_list(2);
dim_re(1) = para_list(3);   %在w位置上降维
dim_re(2) = para_list(4);   %在h位置上降维
tr = para_list(5);

U{1} = randn(w,w);
U{1} = orth(U{1});
U{1} = U{1}(:,(1:dim_re(1)));
P{1} = randn(dim_re(1),w);

U{2}= randn(h,h);
U{2} = orth(U{2});
U{2} = U{2}(:,(1:dim_re(2)));
P{2} = randn(dim_re(2),h);

% Construct Matrix G
% 构建G矩阵
G = updateG(train_all,P,U,tr);


%% Algorithm Start 


obji =1;
for k=1:2
    iter = 1;
    while 1
        % Construct Matrix Q
        %构造Q矩阵
        a = size(U{k},1);
        for i=1:a
            temp_q(i,i) = 1/norm(U{k}(i,:),2);
        end
        Q{k} = temp_q;
        clear temp_q
        
        % construct matrix F_k = Z
        % 构造文中的F_k矩阵。 用Ff表示
        Ff = zeros(size(U{k}*U{k}'));
        for i=1:pen*tr
            if k==1
                Xi_ = P{2}'*U{2}'*train_all(:,:,i);
            else
                Xi_ = (train_all(:,:,i)*U{1}*P{1})';
            end
            Ff = Ff +Xi_'*G(i,i)*W(i,i)*Xi_;
        end
        clear Xi_ 
        
        % construct matrix D_k = Dd
        % 构造文中的D_k矩阵。 用Dd表示
        Dd = zeros(size(U{k}*U{k}'));
        for i=1:pen*tr
            for j=1:pen*tr
                if k==1
                    Xi_ = P{2}'*U{2}'*train_all(:,:,i);
                    Xj = train_all(:,:,j);
                else 
                    Xi_ = (train_all(:,:,i)*U{1}*P{1})';
                    Xj = (train_all(:,:,j))';
                end
                
                Dd = Dd + Xi_'*G(i,j)*W(i,j)*Xj;
            end
        end
       %% v2
        
        A_0 = U{k}'*Ff*U{k} ;
        % 构造Pk，用文中式子(15)
        % Construct P_k, using formula (15)
        P{k} = A_0 \ U{k}' * Dd;
        
        % formula (16) 
        % 式子16的特征分解
        % it's necessary to admit the the formula (16) is wrong,which
        % shouble be (gama(k) * Q{k} * F_k - D_k * D_k) * u =  F_k * u;
        % 有必要说明，因为某种原因，式子16是错误的。
        % 应该是上式。
        I = 1e-5 * eye(size(Ff));
        [eigvec,eigval] = eig(para(k)*(Q{k}-I)*Ff-Dd*Dd');
        [~,idx] = sort(real(diag(eigval)));
        U{k} = eigvec(:,idx(1:dim_re(k)));
        
        % converge analysis
        % 收敛性分析
%         ALL = zeros(size(P{k}'*P{k}));
%         for i=1:pen*tr
%             for j=1:pen*tr
%                 if k==2
%                     Xi_ = train_all(:,:,i)*U{1}*P{1};
%                     Xj_ = train_all(:,:,j)*U{1}*P{1};
%                     Xi = train_all (:,:,i);
%                     Xj = train_all (:,:,j);
%                 else 
%                     Xi_ = P{2}'*U{2}'*train_all(:,:,i)';
%                     Xj_ = P{2}'*U{2}'*train_all(:,:,j)';
%                     Xj = train_all(:,:,j)';
%                     Xi = train_all(:,:,i)';
%                 end
%                 ALL = ALL + P{k}'*U{k}'*(Xi_' * G(i,j,k)*W(i,j) * Xj);
%                 ALL = ALL +  P{k}'*U{k}'*(Xj_' * G(i,j,k)*W(i,j) * Xj_);
%                 ALL = ALL + Xi'* G(i,j,k)*W(i,j)*Xi;
%             end
%         end
%         ALL = ALL + para(k)*P{k}'*U{k}*P{k};
%         obj(iter)=trace(ALL);
%         cver = abs(real(obj(iter))-real(obji))/real(obji);
%         obji = obj(iter);
        iter = iter+1;
        cver = 0;
        
        if ((cver < 10^-3 && iter > 1 )|| iter > 10) , break, end
        G = updateG(train_all,P,U,tr);
    end
end
U1 = U{1};
U2 = U{2};
