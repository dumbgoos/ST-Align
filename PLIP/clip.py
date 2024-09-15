
import torch
import torch.nn.functional as F
import timm
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        """
        Initializes the ProjectionHead module.

        This module projects the input embeddings to a specified projection dimension,
        applies a GELU activation, passes through a fully connected layer,
        applies dropout, adds a residual connection, and finally normalizes the output.

        Parameters:
        embedding_dim (int): Dimension of the input embeddings.
        projection_dim (int, optional): Dimension to project the embeddings to. Default is 256.
        dropout (float, optional): Dropout rate to apply after the fully connected layer. Default is 0.1.
        """
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        """
        Forward pass for the ProjectionHead module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, embedding_dim).

        Returns:
        torch.Tensor: Output tensor with shape (batch_size, projection_dim),
                      after projection, activation, fully connected layer,
                      dropout, residual connection, and layer normalization.
        """
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

class STEncoder(nn.Module):
    def __init__(
        self, 
        model_name="vit_base_patch16_224", 
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    ):
        """
        Initializes the STEncoder module.

        This module creates a Vision Transformer (ViT) model using the timm library.
        The model parameters can be customized to suit different tasks and datasets.

        Parameters:
        model_name (str, optional): Name of the ViT model architecture to use. Default is "vit_base_patch16_224".
        img_size (int, optional): Size of the input images. Default is 224.
        patch_size (int, optional): Size of the patches the image is divided into. Default is 16.
        init_values (float, optional): Initial values for layer normalization. Default is 1e-5.
        num_classes (int, optional): Number of output classes for classification tasks. Set to 0 for feature extraction. Default is 0.
        dynamic_img_size (bool, optional): Allows the model to accept images of varying sizes if set to True. Default is True.
        """
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, 
            img_size=img_size, 
            patch_size=patch_size, 
            init_values=init_values, 
            num_classes=num_classes, 
            dynamic_img_size=dynamic_img_size
        )

    def forward(self, x):
        """
        Forward pass for the STEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
        torch.Tensor: Output tensor from the ViT model, shape depends on the model configuration.
        """
        return self.model(x)


class EXPEncoder(nn.Module):
    def __init__(self,
        d_model=19264,
        layer_nums=12,
        nhead=8,
        mlp_hidden_dim=4096,
        output_dim=1024,
        mlp_layers=3
    ):
        """
        Initializes the EXPEncoder module.

        This module consists of a Transformer encoder followed by a Multi-Layer Perceptron (MLP).
        The Transformer encoder captures contextual information from the input sequence,
        and the MLP projects the encoded features to the desired output dimension.

        Parameters:
        d_model (int, optional): Dimensionality of the model's embedding space. Default is 19264.
        layer_nums (int, optional): Number of layers in the Transformer encoder. Default is 12.
        nhead (int, optional): Number of attention heads in each Transformer encoder layer. Default is 8.
        mlp_hidden_dim (int, optional): Hidden layer size in the MLP. Default is 4096.
        output_dim (int, optional): Output dimensionality of the final MLP layer. Default is 1024.
        mlp_layers (int, optional): Number of layers in the MLP. Default is 3.
        """
        super().__init__()
        layers = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = torch.nn.TransformerEncoder(layers, layer_nums)

        mlp_layers_list = []
        input_dim = d_model
        
        for _ in range(mlp_layers - 1):
            mlp_layers_list.append(nn.Linear(input_dim, mlp_hidden_dim))
            mlp_layers_list.append(nn.ReLU())
            mlp_layers_list.append(nn.LayerNorm(mlp_hidden_dim))
            input_dim = mlp_hidden_dim
        
        mlp_layers_list.append(nn.Linear(input_dim, output_dim))
        mlp_layers_list.append(nn.LayerNorm(output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers_list)

    def forward(self, x):
        """
        Forward pass for the EXPEncoder module.

        Parameters:
        x (torch.Tensor): Input tensor of shape (sequence_length, batch_size, d_model).

        Returns:
        torch.Tensor: Output tensor of shape (sequence_length, batch_size, output_dim),
                      after processing through the Transformer encoder and MLP layers.
        """
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=768,
        text_embedding=1024,
    ):
        """
        Initializes the CLIPModel module.

        This model combines an image encoder and a text encoder, projects their outputs into a shared embedding space,
        and computes a contrastive loss to align image and text representations.

        Parameters:
        temperature (float, optional): Temperature parameter for scaling logits. Default is 1.0.
        image_embedding (int, optional): Dimensionality of the image embedding space. Default is 768.
        text_embedding (int, optional): Dimensionality of the text embedding space. Default is 1024.
        """
        super().__init__()
        self.image_encoder = STEncoder()
        self.text_encoder = EXPEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        """
        Forward pass for the CLIPModel.

        Parameters:
        batch (dict): A dictionary containing:
            - "image" (torch.Tensor): Batch of images with shape (batch_size, channels, height, width).
            - "input_ids" (torch.Tensor): Tokenized text input IDs with shape (batch_size, sequence_length).
            - "attention_mask" (torch.Tensor): Attention masks for text inputs with shape (batch_size, sequence_length).

        Returns:
        torch.Tensor: Scalar loss value computed from the contrastive loss between image and text embeddings.
        """
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            batch["input_ids"], batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / (2 * self.temperature), dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0  # shape: (batch_size,)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    """
    Computes the cross-entropy loss between predictions and targets.

    Parameters:
    preds (torch.Tensor): Prediction logits with shape (batch_size, num_classes).
    targets (torch.Tensor): Target probabilities with shape (batch_size, num_classes).
    reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean'. Default is 'none'.

    Returns:
    torch.Tensor: The computed loss. If reduction is 'none', returns a tensor of shape (batch_size,);
                  if 'mean', returns a scalar tensor.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
