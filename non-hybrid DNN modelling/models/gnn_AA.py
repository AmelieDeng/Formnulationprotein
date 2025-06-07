import itertools

import torch
from torch.nn import Sigmoid, ReLU
from models.egnn_clean import egnn_clean as eg
import net_utils
import torch.nn as nn
import torch.nn.functional as F

class ResidualFCBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(ResidualFCBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.tanh = nn.Tanh()

        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        #out = F.relu(out)
        out = self.tanh(out)     

        out = self.fc2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out += identity
        #out = F.relu(out)
        out = self.tanh(out)  
         
        return out



class GCN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GCN, self).__init__()

        num_class = kwargs['num_classes']
        input_features_size = kwargs['input_features_size']
        hidden_channels = kwargs['hidden']
        edge_features = kwargs['edge_features']

        output_features_size = 160

        num_egnn_layers = kwargs['egnn_layers']

        self.edge_type = kwargs['edge_type']
        self.num_layers = kwargs['layers']
        self.device = kwargs['device']
        
        forminput = kwargs['formnums']
        form_FC_dim1 = 320
        form_FC_dim2 = 160
        form_FC_dim3 = 80
        
        AAinput = kwargs['AAnums']
        AA_FC_dim1 = 160
        AA_FC_dim2 = 120
        AA_FC_dim3 = 30

        #egnn blocks
        self.egnn_1 = eg.EGNN(in_node_nf=input_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=output_features_size,
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True,
                              device = self.device)

        self.egnn_2 = eg.EGNN(in_node_nf=output_features_size,
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(output_features_size / 2),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True,
                              device = self.device)

        self.egnn_3 = eg.EGNN(in_node_nf=int(output_features_size / 2),
                              hidden_nf=hidden_channels,
                              n_layers=num_egnn_layers,
                              out_node_nf=int(output_features_size / 4),
                              in_edge_nf=edge_features,
                              attention=True,
                              normalize=False,
                              tanh=True,
                              device = self.device)

        self.bnrelu1 = net_utils.BNormRelu(output_features_size)
        self.bnrelu2 = net_utils.BNormRelu(int(output_features_size / 2))
        self.bnrelu3 = net_utils.BNormRelu(int(output_features_size / 4))

        #formulation blocks        
        self.fc_form1 = net_utils.FC(forminput, form_FC_dim1, relu=False, bnorm=False)
        self.fc_form2 = net_utils.FC(form_FC_dim1, form_FC_dim2, relu=False, bnorm=False)
        self.fc_form3 = net_utils.FC(form_FC_dim2, form_FC_dim3, relu=False, bnorm=False)  
        self.fc_form4 = net_utils.FC(form_FC_dim3, form_FC_dim3, relu=False, bnorm=False)  
        
        ####formulation layer definition
        self.bnrelu_form1 = net_utils.BNormRelu(int(form_FC_dim1))
        self.bnrelu_form2 = net_utils.BNormRelu(int(form_FC_dim2))
        self.bnrelu_form3 = net_utils.BNormRelu(int(form_FC_dim3))    
        self.bnrelu_form4 = net_utils.BNormRelu(int(form_FC_dim3))            

        #AA blocks        
        self.fc_AA1 = net_utils.FC(AAinput, AA_FC_dim1, relu=False, bnorm=False)
        self.fc_AA2 = net_utils.FC(AA_FC_dim1, AA_FC_dim2, relu=False, bnorm=False)
        self.fc_AA3 = net_utils.FC(AA_FC_dim2, AA_FC_dim3, relu=False, bnorm=False)  
        self.fc_AA4 = net_utils.FC(AA_FC_dim3, AA_FC_dim3, relu=False, bnorm=False)  
        
        ####AA layer definition
        self.bnrelu_AA1 = net_utils.BNormRelu(int(AA_FC_dim1))
        self.bnrelu_AA2 = net_utils.BNormRelu(int(AA_FC_dim2))
        self.bnrelu_AA3 = net_utils.BNormRelu(int(AA_FC_dim3))    
        self.bnrelu_AA4 = net_utils.BNormRelu(int(AA_FC_dim3)) 


        #downsteam blocks    
        # self.fc1 = net_utils.FC(output_features_size + int(output_features_size / 2) + int(output_features_size / 4) + form_FC_dim2,
        #                         int(output_features_size/2), relu=True, bnorm=True)      
        # self.fc2 = net_utils.FC(int(output_features_size/2), int(output_features_size/8), relu=True, bnorm=True) 
        # self.fc3 = net_utils.FC(int(output_features_size/8), int(output_features_size/8), relu=True, bnorm=True)
        # self.fc4 = net_utils.FC(int(output_features_size/8), int(output_features_size/8), relu=True, bnorm=True)
        
        #res blocks 
        self.reslayer1 = ResidualFCBlock(output_features_size + int(output_features_size / 2) + int(output_features_size / 4) + form_FC_dim3 + AA_FC_dim3, 125, 32)
        self.reslayer2 = ResidualFCBlock(32,32,32)      
        self.reslayer3 = ResidualFCBlock(32,32,32)
        self.reslayer4 = ResidualFCBlock(32,32,32)
        
        self.final = net_utils.FC(32, num_class, relu=False, bnorm=False)           
        # self.relu = ReLU()
        #self.sig = Sigmoid()

    def forward_once(self, data):

        x_res, edge_index, x_batch, x_pos,x_form_one, x_AA, edge_attr = data['atoms'].embedding_features_per_residue, \
            data[self.edge_type].edge_index, \
            data['atoms'].batch, \
            data['atoms'].pos, data['atoms'].form_one,data['atoms'].AA, \
            data[self.edge_type].edge_attr
        
        #print('protein', data[self.edge_type])
        #print('atom', data['atoms'].pos.shape)            
        #print('atom', data['atoms'].embedding_features_per_residue.shape)  
        
        output_res, pre_pos_res = self.egnn_1(h=x_res,
                                              x=x_pos.float(),
                                              edges=edge_index,
                                              edge_attr=edge_attr)

        output_res_2, pre_pos_res_2 = self.egnn_2(h=output_res,
                                                  x=pre_pos_res.float(),
                                                  edges=edge_index,
                                                  edge_attr=edge_attr)
        output_res_3, pre_pos_seq_3 = self.egnn_3(h=output_res_2,
                                                  x=pre_pos_res_2.float(),
                                                  edges=edge_index,
                                                  edge_attr=edge_attr)

        output_res = net_utils.get_pool(pool_type='mean')(output_res, x_batch)
        output_res = self.bnrelu1(output_res)

        output_res_2 = net_utils.get_pool(pool_type='mean')(output_res_2, x_batch)
        output_res_2 = self.bnrelu2(output_res_2)

        output_res_3 = net_utils.get_pool(pool_type='mean')(output_res_3, x_batch)
        output_res_3 = self.bnrelu3(output_res_3)

######formulation data                
        output_form = self.fc_form1(x_form_one)
        output_form = self.bnrelu_form1(output_form)
        output_form = self.fc_form2(output_form)
        output_form = self.bnrelu_form2(output_form)
        output_form = self.fc_form3(output_form)
        output_form = self.bnrelu_form3(output_form)
        output_form = self.fc_form4(output_form)
        output_form = self.bnrelu_form4(output_form)
        
######AA data                
        output_AA = self.fc_AA1(x_AA)
        output_AA = self.bnrelu_AA1(output_AA)
        output_AA = self.fc_AA2(output_AA)
        output_AA = self.bnrelu_AA2(output_AA)
        output_AA = self.fc_AA3(output_AA)
        output_AA = self.bnrelu_AA3(output_AA)
        output_AA = self.fc_AA4(output_AA)
        output_AA = self.bnrelu_AA4(output_AA)        

        output = torch.cat([output_res, output_form, output_AA, output_res_2, output_res_3], 1)
 
        return output

    def forward(self, data):
        passes = []

        for i in range(self.num_layers):
            passes.append(self.forward_once(data))

        x = torch.cat(passes, 1)

        x = self.reslayer1(x)
        x = self.reslayer2(x)
        x = self.reslayer3(x)
        x = self.reslayer4(x)        
        x = self.final(x)
        # x = self.sig(x)

        return x
