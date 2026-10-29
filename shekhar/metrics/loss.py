import torch
import torch.nn as nn

class PatchLoss(nn.Module):
    def __init__(self, config):
        super(PatchLoss, self).__init__()
        self.config = config
        self.device = config.experiment.device
        self.ignore_label = config.train.ignore_label


    def compute_loss(self, model_output, label):
        """
        Compute the adaptive loss function
        """

        #print(model_output.shape,true_labels.shape)
        #print(model_output.argmax(dim=1).shape)
        ce_loss = nn.CrossEntropyLoss(reduction="none",
                                      ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss_map = ce_loss(model_output, label.long())  # Compute loss for all pixels
        #print(f'loss map: {loss_map.shape}')
              
      
        # Get correctly classified and misclassified pixel sets
        predict = torch.argmax(model_output, 1).float() + 1
        target = label.float() + 1
        target[target>=255] = 0
        # print(predict.dtype,predict.shape,target.dtype,target.shape)
        # temp1 = (predict == target).float()
        # temp2 = (target>0).float()
        # print(temp1.dtype,temp2.dtype)
        correct_mask = (predict == target)*(target > 0)
        incorrect_mask = (predict != target)*(target > 0)  # Opposite of correctly classified
        #print(f'Correct mask: {correct_mask.shape}')  
        # Compute separate loss terms
        #
        # GUARDED DENOMINATORS. All three divisions below are 0/0 whenever a
        # frame has no pixel in the relevant set, and 0/0 is a NaN forward with
        # a non-finite backward. Two ways that happens here, both routine:
        #
        #   - correct_mask.sum() == 0: once the patch is strong enough that
        #     nothing in the frame is classified correctly any more.
        #   - num_total == 0: a crop containing no labelled pixels at all.
        #     With rand_scale up to 2.1 (base_dataset.gen_sample) a 1024x1024
        #     crop can land entirely inside void -- ego-vehicle hood or border
        #     -- and the patch footprint is set to ignore_label on top of that.
        #
        # The unguarded version produced `Loss: nan` at every logged iteration
        # of a 1-epoch run while leaving the patch itself finite: correct_mask
        # is bool, so .sum() is a long tensor and autograd treats the
        # denominator as a constant -- NaN forward, inf backward, sign(inf)=+1,
        # so the patch kept taking meaningless +epsilon steps.
        #
        # clamp(min=1) rather than + eps. Both remove the NaN, but with a tiny
        # eps the division carries a 1/eps gradient multiplier (1e10 for
        # 1e-10); that is harmless only while the numerator's gradient is
        # exactly zero, which is true in the fully degenerate case and not
        # something worth relying on. clamp keeps the scale at 1 and matches
        # the convention used in patchreach's own _reduce().
        n_correct = correct_mask.sum().clamp(min=1)
        n_incorrect = incorrect_mask.sum().clamp(min=1)

        loss_correct = (loss_map * correct_mask).sum()/n_correct  # Loss on correctly classified pixels
        loss_incorrect = (loss_map * incorrect_mask).sum()/n_incorrect  # Loss on already misclassified pixels

        # Compute adaptive balancing factor
        num_correct = correct_mask.sum()
        num_total = (target != 0).sum().clamp(min=1)
        gamma = num_correct / num_total  # Avoid division by zero

        # Final adaptive loss
        loss = gamma * loss_correct + (1 - gamma) * loss_incorrect
        # print(f'Gamma:{gamma}')
        # print(f'loss correct:{loss_correct}')
        # print(f'loss incorrect: {loss_incorrect}')
        #return loss
        return loss_correct 


    def compute_loss_direct(self, model_output, label):
        """
        Compute the adaptive loss function
        """
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.config.train.ignore_label)  # Per-pixel loss
        loss = ce_loss(model_output, label.long())  # Compute loss for all pixels
        return loss 