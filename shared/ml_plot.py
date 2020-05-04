import matplotlib.pyplot as plt
import numpy as np

def get_merged_mask(label,debris):
    label = np.load(label).astype(int)
    label = np.where((label==0)|(label==1), label^1, label)
    debris = np.load(debris).astype(int)
    debris = np.where((debris==0)|(debris==1), debris^1, debris)
    merged = np.add(label, debris)
    merged = np.where((merged==0)|(merged==1), merged^1, merged)
    return merged

def pixel_acc(true_label, predicted_label):
    total_pix = true_label.shape[0]*true_label.shape[1]
    true_pred = 0
    for i in range(true_label.shape[0]):
        for j in range(true_label.shape[1]):
            if true_label[i][j] == predicted_label[i][j]:
                true_pred += 1
    perc = true_pred/total_pix*100
    print("Pixel Accuracy = "+str(perc))

if __name__=="__main__":
    tile_name = "LE07_140041_20051012_slice_71"

    image_name = "../data/slices/img_"+tile_name+".npy"
    label = "../data/slices/mask_"+tile_name+".npy"
    debris_label = "../data/slices/actual_debris_mask_"+tile_name+".npy"

    output_svm_linear = "./inference_data/svm_linear_output.npy"
    output_svm_rbf = "./inference_data/svm_rbf_output.npy"
    output_decision_tree = "./inference_data/decision_tree_output.npy"
    output_mlp = "./inference_data/mlp_output.npy"

    image = np.load(image_name)
    image_rgb = image[:,:,[0,2,1]].astype(int)
    image_542 = image[:,:,[4,3,1]].astype(int)

    image_dt = np.load(output_decision_tree).astype(int)
    image_mlp = np.load(output_mlp).astype(int)
    image_svm_rbf = np.load(output_svm_rbf).astype(int)
    image_svm_linear = np.load(output_svm_linear).astype(int)

    true_label = get_merged_mask(label,debris_label)
    plt.title(tile_name)

    plt.subplot(331)
    plt.title("RGB Image")
    plt.imshow(image_rgb,cmap="brg")
    # plt.imsave("./raw_image.png", image_rgb)

    plt.subplot(332)
    plt.title("Image (channel 542)")
    plt.imshow(image_542,cmap="brg")
    # plt.imsave("./542_image.png", image_542)

    plt.subplot(333)
    plt.title("True Label")
    plt.imshow(true_label)
    # plt.imsave("./true_label.png", true_label)

    plt.subplot(334)
    plt.title("Decision Tree")
    plt.imshow(image_dt)

    plt.subplot(335)
    plt.title("SVM Linear")
    plt.imshow(image_svm_linear)
    
    plt.subplot(336)
    plt.title("SVM RBF")
    plt.imshow(image_svm_rbf)

    plt.subplot(337)
    plt.title("MLP")
    plt.imshow(image_mlp)

    plt.subplot(338)
    plt.title("MLP after CRF")
    
    plt.subplot(339)
    plt.title("Legend")

    legend = ["Clean Ice","Debris Glaciers","Background"]
    colors = ["indigo", "yellow", "green"]

    f = lambda m,c: plt.plot([],[], marker=m, color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(3)]
    
    labels = legend
    legend = plt.legend(handles, labels, loc=10, framealpha=1, frameon=None)
    fig  = legend.figure
    fig.patch.set_visible(False)
    fig.canvas.draw()

    plt.savefig("./summarized_plot_"+tile_name)
    plt.show()