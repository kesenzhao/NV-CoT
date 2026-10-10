import unittest
import torch
from policy import JointVisualRefiner,visual_context,clipped_loss,group_advantages
from visual_refiner import VisualBoxRefiner,build_visual_context

class Tests(unittest.TestCase):
    def test_pool_equivalence_and_gradient(self):
        torch.manual_seed(4)
        tokens=torch.randn(64,8,requires_grad=True);grid=torch.tensor([[1,16,16]])
        box=torch.tensor([.1,.2,.7,.8])
        old=build_visual_context(tokens,grid,0,box)
        new=visual_context(tokens,grid,box)
        for key,value in zip(('tokens','xy','branch','valid'),new):
            torch.testing.assert_close(old[key],value,atol=0,rtol=0)
        new[0].sum().backward();self.assertGreater(float(tokens.grad.abs().sum()),0)

    def test_head_forward_equivalence_and_gradient(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        old=VisualBoxRefiner(8,8,width=8,heads=2,layers=2).eval()
        with torch.no_grad():old.output[-1].weight.normal_(0,.1)
        new=JointVisualRefiner(8,8,width=8,heads=2,layers=2).eval();new.load_state_dict(old.state_dict())
        h=torch.randn(1,8,requires_grad=True);tokens=torch.randn(1,16,8,requires_grad=True)
        args=(h,torch.tensor([[.1,.1,.8,.8]]),torch.tensor([[640.,480.]]),tokens,
              torch.rand(1,16,2),torch.zeros(1,16,dtype=torch.long),torch.ones(1,16,dtype=torch.bool))
        out=new(*args);torch.testing.assert_close(out,old(*args),atol=0,rtol=0)
        out.sum().backward();self.assertGreater(float(h.grad.abs().sum()),0);self.assertGreater(float(tokens.grad.abs().sum()),0)

    def test_on_policy_loss_has_gradient(self):
        logits=torch.tensor([.1,.2,.3],requires_grad=True)
        new=torch.log_softmax(logits,-1)[0]
        loss=clipped_loss(new,new.detach(),torch.tensor(1.));loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()),0)
        advantage=group_advantages([0.,.8,0.,.8])
        torch.testing.assert_close(advantage,torch.tensor([-1.,1.,-1.,1.]))
        self.assertEqual(float(group_advantages([.8]*4).abs().sum()),0)

if __name__=='__main__':unittest.main()
