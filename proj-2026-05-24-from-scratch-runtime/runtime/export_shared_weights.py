import os, torch, itertools
from finalize_ref import load_model
ART="artifacts"
m=load_model()
enc=m.encoder
cmap={}
for name,t in itertools.chain(enc.named_parameters(), enc.named_buffers()):
    cmap["e."+name]=t.detach().cpu()
torch.save(cmap, os.path.join(ART,"finalize_shared_weights.pt"))
print("saved", len(cmap), "weights ->", os.path.join(ART,"finalize_shared_weights.pt"))
