function G = updateG(train_all,P,U,tr)
global pen
G = zeros(pen*tr,pen*tr);
for i=1:pen*tr
    for j=1:pen*tr
        if(i==j)
            G(i,j) = 10^-6;
        else
            G(i,j) =  createG(train_all(:,:,i),train_all(:,:,j),P,U);
        end
    end
end