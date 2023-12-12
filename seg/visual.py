import cv2
def visualized_gt(img, gt):
    img = img.cpu()
    gt = gt.cpu()
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    output_img = np.zeros((512, 512, 3), dtype=np.uint8)

    output_img[gt[0] == 1] = [0, 0, 255] 

    cv2.imwrite('img.png', img_np)
    cv2.imwrite('gt.png', output_img)

    return 0