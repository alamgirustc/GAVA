import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoAttention(nn.Module):
    def __init__(self, embed_dim):
        super(GeoAttention, self).__init__()
        self.attention = nn.Linear(embed_dim, 1, bias=False)
        self.feature_fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, att_feats, geo_feats_transformed):
        fused_feats = torch.cat((att_feats, geo_feats_transformed), dim=-1)
        fused_feats = self.feature_fusion(fused_feats)
        att_weights = F.softmax(self.attention(fused_feats), dim=1)
        weighted_geo_feats = att_weights * geo_feats_transformed
        return att_feats + weighted_geo_feats

