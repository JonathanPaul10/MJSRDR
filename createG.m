function G =  createG(Xi,Xj,P,U)
G = 0;
a = size(Xi,1);
F = Xi-P{2}'*U{2}'*Xj*U{1}*P{1};
for i=1:a
    G = G + 1 / (2*norm(F(i,:),2));
end
