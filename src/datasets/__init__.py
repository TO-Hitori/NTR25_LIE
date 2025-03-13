# from .NTIRE_Dataset import NTIRE_LIE_Dataset





    # batch_sizes = [16, 8, 4, 2]
    # patch_sizes = [64, 128, 512, 1024]
    # # steps = np.array(steps)
    # # steps = np.array(steps)
    # device = torch.device('cuda')
    # sample = torch.randn(batch_sizes[0], 3, 3000, 3000).to(device)
    #
    # for i in range(0, 120):
    #     # comp = (i < steps).nonzero()[0]
    #     # state_idx = len(steps) - 1 if len(comp) == 0 else comp[0]
    #     #
    #     # batch_size = batch_sizes[state_idx]
    #     # patch_size = patch_sizes[state_idx]
    #     #
    #     # batch_idx = torch.randperm(batch_sizes[0])[:batch_size]
    #     batch_idx, patch_size = get_batch_idx_patch_size(i, steps, batch_sizes, patch_sizes)
    #     sample_i = sample[batch_idx][:, :, :patch_size, :patch_size]
    #
    #     print(sample_i.shape)

