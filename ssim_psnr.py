from pytorch_msssim import ssim

# SSIM
ssim_score += ssim(batch_features.view(
(-1, 3, 256, 256)), outputs[0].view((-1, 3, 256, 256)))

# PSNR
mse = torch.mean((batch_features.view((-1, 3, 256, 256)
                                        ) - outputs[0].view((-1, 3, 256, 256))) ** 2)
psnr += 20 * torch.log10(255.0 / torch.sqrt(mse))