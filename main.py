import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_loader import get_uci_har_data
from condtsc_modules import Dualmodel, KCenter, DataTransform_TD


def grad_match_loss(real_loss, fake_loss, network_part):
    # comparing gradients from real and synthetic batches

    real_grads = torch.autograd.grad(real_loss, network_part.parameters(), create_graph=True)
    synth_grads = torch.autograd.grad(fake_loss, network_part.parameters(), create_graph=True)
    total = 0

    for r_grad, f_grad in zip(real_grads, synth_grads):
        error = (r_grad - f_grad) ** 2
        total += torch.sum(error)
    return total


class args:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # setting teacher architecture to cnnbn
        self.model = "CNNBN"
        # params based on human activities dataset- should be changed when testing with other dataset
        self.channel = 9
        self.num_classes = 6
        self.time_step = 128
        # 4% of human activities dataset
        self.ipc = 50
        # params from paper
        self.jitter_ratio = 0.1
        self.jitter_scale_ratio = 0.1
        self.max_seg = 5
        self.dual = 1


def synth_data(X_real, y_real, args):

    print("\nStarting synthetic dataset gen")

    class DataWrapper:
        def __init__(self, X, y):
            self.feat_train = X
            self.labels_train = y            # index tracking
            self.idx_train = np.arange(len(y))

    wrapped_data = DataWrapper(X_real, y_real)

    kcenter = KCenter(wrapped_data, args, device=args.device)

    X_real_tensor = torch.tensor(X_real, dtype=torch.float32).to(args.device)
    X_real_flat = X_real_tensor.view(X_real_tensor.shape[0], -1)
    selected_idx = kcenter.select(X_real_flat, inductive=True)

    # initialize synthetic data
    S_x = torch.tensor(X_real[selected_idx], dtype=torch.float32, requires_grad=True, device=args.device)
    S_y = (kcenter.labels_syn.clone().detach().to(dtype=torch.long, device=args.device))

    # setitng optimizer and loss func
    syn_optimizer = optim.Adam([S_x], lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # teacher model
    teacher_model = Dualmodel(args).to(args.device)

    model_optimizer = optim.Adam(teacher_model.parameters(),lr=0.01)

    print("Running dual domain matching\n")

    for epoch in range(1000):
        teacher_model.train()
        running_loss = 0
        syn_optimizer.zero_grad()
        # updating synthetic data
        for class_id in range(args.num_classes):
            class_indices = np.where(y_real == class_id)[0]
            if len(class_indices) == 0:
                continue
            # sample real data batch
            batch_size = min(128, len(class_indices))

            sampled_idx = np.random.choice(class_indices, batch_size, replace=False)

            real_batch_x = torch.tensor(X_real[sampled_idx], dtype=torch.float32).to(args.device)

            real_batch_y = torch.tensor(y_real[sampled_idx], dtype=torch.long).to(args.device)

            # synthetic class batch
            syn_class_idx = torch.where(S_y == class_id)[0]
            syn_batch_x = S_x[syn_class_idx]
            syn_batch_y = S_y[syn_class_idx]

            # differentiable augmentations
            real_aug = DataTransform_TD(real_batch_x, args)
            syn_aug = DataTransform_TD(syn_batch_x, args)
            real_output, _, _ = teacher_model(real_aug)
            real_loss = criterion(real_output, real_batch_y)
            syn_output, _, _ = teacher_model(syn_aug)
            syn_loss = criterion(syn_output, syn_batch_y)

            # gradient matching
            match_loss = grad_match_loss(real_loss, syn_loss, teacher_model.mlp)
            match_loss.backward()
            running_loss += match_loss.item()

        syn_optimizer.step()

        # update teacher network
        model_optimizer.zero_grad()
        detached_syn = S_x.detach()

        detached_syn_aug = DataTransform_TD(detached_syn,args)

        syn_eval_output, _, _ = teacher_model(detached_syn_aug)
        syn_eval_loss = criterion(syn_eval_output, S_y)
        syn_eval_loss.backward()
        model_optimizer.step()

        # reset network to avoid overfitting every 100 epochs
        if (epoch + 1) % 100 == 0:
            avg_loss = running_loss / args.num_classes

            print(f"Epoch {epoch+1} | " f"avg matching loss = {avg_loss:.4f}")

            teacher_model = Dualmodel(args).to(args.device)
            model_optimizer = optim.Adam(teacher_model.parameters(), lr=0.01)

            # check if resetting is good approach

    print("\nSynthetic dataset gen completed.")

    return (S_x.detach().cpu().numpy(), S_y.cpu().numpy())



def plot_cross_architecture(results_dict, baseline_results):

    plt.figure(figsize=(10, 6))

    arch_names = list(results_dict.keys())
    arch_scores = list(results_dict.values())

    bars = plt.bar(
        arch_names,
        arch_scores,
        color="skyblue",
        edgecolor="black",
        label="Deep Models (Condensed Data)"
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            fontweight="bold"
        )

    line_colors = ["red", "green", "orange"]

    for idx, (name, score) in enumerate(baseline_results.items()):
        plt.axhline(y=score, color=line_colors[idx], linestyle="--", label=f"{name} (Full Dataset)")

    plt.ylim(0, 100)
    plt.xlabel("Evaluation Architecture")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Cross-Architecture Generalization Capability",
        fontweight="bold"
    )
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )
    plt.legend(loc="lower right")
    save_name = "cross_arch_results.png"
    plt.savefig(
        save_name,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved graph as {save_name}")

# main func

def run_simulations():

    args = args()
    args.channel = 9
    args.num_classes = 6
    args.time_step = 128
    args.ipc = 50

    X_train, y_train, X_test, y_test = get_uci_har_data("UCI HAR Dataset")

    print("dataset loaded successfully")

    # teacher model set as CNN
    args.model = "CNNBN"

    S_x, S_y = synth_data(X_train,y_train,args)

    print("\nChecking baseline performance on full dataset")

    X_train_flat = X_train.reshape(X_train.shape[0],-1)

    X_test_flat = X_test.reshape(X_test.shape[0],-1)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50)
    rf_model.fit(X_train_flat, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_flat)) * 100

    # SVM
    svm_model = SVC(kernel="rbf")
    svm_model.fit(X_train_flat, y_train)
    svm_acc = accuracy_score( y_test,svm_model.predict(X_test_flat) ) * 100

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_flat, y_train)
    knn_acc = accuracy_score(y_test,knn_model.predict(X_test_flat)) * 100

    baseline_results = {
        "SVM": svm_acc,
        "Random Forest": rf_acc,
        "KNN": knn_acc
    }

    print("\nBaseline Results:")
    print(baseline_results)

    # cross architecture evaluation

    architectures = ["CNNBN","LSTM","MLP","TCN"]
    cross_arch_results = {}

    S_x_t = torch.tensor(S_x,dtype=torch.float32).to(args.device)
    S_y_t = torch.tensor(S_y,dtype=torch.long).to(args.device)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(args.device)

    criterion = nn.CrossEntropyLoss()

    # training separate models on condensed dataset
    for arch_name in architectures:

        print(f"Training model {arch_name}")
        local_args = copy.deepcopy(args)
        local_args.model = arch_name
        eval_model = Dualmodel(local_args).to(args.device)
        eval_optimizer = optim.Adam(eval_model.parameters(), lr=0.01)

        for epoch in range(200):
            eval_model.train()
            eval_optimizer.zero_grad()
            outputs, _, _ = eval_model(S_x_t)
            loss = criterion(outputs, S_y_t)
            loss.backward()
            eval_optimizer.step()

        eval_model.eval()

        with torch.no_grad():
            test_outputs, _, _ = eval_model(X_test_t)
            preds = torch.argmax(test_outputs,dim=1).cpu().numpy()
            acc = accuracy_score(y_test, preds) * 100
            cross_arch_results[arch_name] = acc
            print(f"{arch_name} Accuracy: {acc:.2f}%")

    plot_cross_architecture(cross_arch_results, baseline_results)
    print("done")


if __name__ == "__main__":
    run_simulations()
