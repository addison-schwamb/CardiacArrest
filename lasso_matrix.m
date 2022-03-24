function w = lasso_matrix(mat,bases,err,idx,dim)
    
    w = zeros(size(mat,dim),length(bases));
    if dim == 2
        mat = mat.';
    end

    for i=1:size(mat,1)
        [l,FitInfo] = lasso(bases,mat(i,idx));
        MSE_below_err = FitInfo.MSE(FitInfo.MSE<err);
        w(i,:) = l(:,length(MSE_below_err)).';
    end

    if dim==2
        w = w.';
    end
end