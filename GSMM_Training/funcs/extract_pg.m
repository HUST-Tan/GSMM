function PG = extract_pg( subs_Batch, O_Img_Batch, NL_mat_Batch )

        nlsp      = size(NL_mat_Batch,1);
        im_PG_num = size(NL_mat_Batch,2);
        im_batch  = size(O_Img_Batch,3);
        
        CurPat_Batch = O_Img_Batch(subs_Batch(:));
        CurPat_Batch = reshape(CurPat_Batch, size(subs_Batch));
        
        NL_mat_Batch = reshape(NL_mat_Batch, nlsp*im_PG_num, im_batch);
        NL_mat_Batch = NL_mat_Batch + (0 : size(subs_Batch,2) : size(subs_Batch,2)*(im_batch-1));
        
        PG = CurPat_Batch(:, NL_mat_Batch(:));
        PG = reshape(PG, size(PG,1)*nlsp, size(PG,2)/nlsp);
        
end

