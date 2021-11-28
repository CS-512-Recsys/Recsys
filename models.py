class NCF(nn.Module):
    def __init__(self,user_sz,item_sz,embd_sz,dropout_fac,min_r=0.0,max_r=5.0,alpha=0.5,with_variable_alpha=False):
        super().__init__()
        self.dropout_fac = dropout_fac
        self.user_embd_mtrx = nn.Embedding(user_sz,embd_sz)
        self.item_embd_mtrx = nn.Embedding(item_sz,embd_sz)
        #bias = torch.zeros(size=(user_sz, 1), requires_grad=True)
        self.h =  nn.Linear(embd_sz,1)
        self.fst_lyr = nn.Linear(embd_sz*2,embd_sz)
        self.snd_lyr = nn.Linear(embd_sz,embd_sz//2)
        self.thrd_lyr = nn.Linear(embd_sz//2,embd_sz//4)
        self.out_lyr = nn.Linear(embd_sz//4,1)
        self.alpha = torch.tensor(alpha)
        self.min_r,self.max_r = min_r,max_r
        if with_variable_alpha:
            self.alpha = torch.tensor(alpha,requires_grad=True)
    def forward(self,x):
        user_emd = self.user_embd_mtrx(x[0])
        item_emd = self.item_embd_mtrx(x[-1])
        #hadamard-product
        gmf = user_emd*item_emd
        gmf = self.h(gmf)
        mlp = torch.cat([user_emd,item_emd],dim=-1)
        mlp = self.out_lyr(self.thrd_lyr(self.snd_lyr(F.dropout(self.fst_lyr(mlp),p=self.dropout_fac))))
        fac = torch.clip(self.alpha,min=0.0,max=1.0)
        out = fac*gmf+ (1-fac)*mlp
        out = torch.clip(out,min=self.min_r,max=self.max_r)
        return out