import string_utils
import torch
import torch.nn as nn
from torch.autograd import Variable

class PsychLoss(nn.Module):
    def __init__(self):
        super(PsychLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
        # we don't care about char for this, only if it got it right or wrong
        # self.idx_to_char = idx_to_char
        # self.char_to_idx = char_to_idx
        # self.verbose = verbose

    def forward(self, preds, labels, preds_size, label_lengths, psych):
        loss = self.criterion(preds, labels).cuda()
        index = 0

        # then we just do OUR psych stuff on this, not hte label processing
        # lbl = labels.detach().cpu().numpy()
        lbl = labels.data.cpu().numpy()
        lbl_len = label_lengths.data.cpu().numpy()
        output_batch = preds.permute(1, 0, 2)
        # out = output_batch.detach().cpu().numpy()
        out = output_batch.data.cpu().numpy()
        cer = torch.zeros(loss.shape)

        # loss and psych are not modified here, just the cer[j]
        # error rate at a sample

        # for sample in logits, 
        #       apply psych penalty 

        for j in range(out.shape[0]):
            logits = out[j, ...]
            pred, raw_pred = string_utils.naive_decode(logits)
            # pred_str = string_utils.label2str(pred, self.idx_to_char, False)
            # gt_str = string_utils.label2str(lbl[index:lbl_len[j] + index], self.idx_to_char, False)
            index += lbl_len[j]
            # oh this is just editdistance.
            # we need our eror metric instead bb
            # cer[j] = error_rates.cer(gt_str, pred_str)


            # TODO: compare this at the actual value
            cer[j] = my_error_fn(pred,label)
            # if self.verbose or psych[j] > 6000:
            #     print(psych[j])
        cer = Variable(cer, requires_grad = True).cuda()
        loss = loss + (psych * cer)
        return torch.sum(loss)