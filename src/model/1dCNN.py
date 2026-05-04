import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScale1DCNN(nn.Module):
    def __init__(self, input_feature_dim=3955):
        """
        논문에서 설명된 다중 스케일 1D-CNN 모델을 구현합니다.

        Args:
            input_feature_dim (int): 입력 특징 벡터의 길이 (예: GLCM, LBP, ResNet50 특징 결합 후의 차원).
                                     논문에서는 3955로 언급되었습니다.
        """
        super(MultiScale1DCNN, self).__init__()

        # ============================================
        # 경로 1: 커널 크기 3 (Narrower scale)
        # ============================================
        self.conv1_narrow = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # padding=1 to keep length
        self.mp1_narrow = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_narrow = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_narrow = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4_narrow = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.mp2_narrow = nn.MaxPool1d(kernel_size=2, stride=2)

        # FC layers after the convolutional layers for the narrow path
        # Calculate the output size after conv and pooling layers to determine FC input size
        # This requires knowing the exact sequence length after all conv/pool ops.
        # Let's assume for now the sequence length can be tracked or we can use adaptive pooling.
        # For simplicity, we'll use a placeholder and then calculate it properly or use AdaptiveAvgPool1d if needed.
        # Given the max pooling with stride 2 twice, the sequence length is roughly divided by 4.
        # input_feature_dim -&gt; (conv_out_len) -&gt; mp1_out_len -&gt; (conv_out_len) * 2 -&gt; mp2_out_len
        # Without explicit padding and stride info for all convs, let's trace it:
        # input_feature_dim
        # Conv1 (k=3, p=1, s=1) -&gt; len remains input_feature_dim
        # MP1 (k=2, s=2) -&gt; len becomes input_feature_dim / 2
        # Conv2 (k=3, p=1, s=1) -&gt; len remains input_feature_dim / 2
        # Conv3 (k=3, p=1, s=1) -&gt; len remains input_feature_dim / 2
        # Conv4 (k=3, p=1, s=1) -&gt; len remains input_feature_dim / 2
        # MP2 (k=2, s=2) -&gt; len becomes (input_feature_dim / 2) / 2 = input_feature_dim / 4
        # The output of this path will be (batch_size, 256, input_feature_dim / 4)

        # To avoid manual calculation of sequence length, we can use AdaptiveMaxPool1d after the last conv layer before flattening.
        # However, the diagram shows specific FC layers *after* the last pooling.
        # Let's assume the sequence length after mp2_narrow is L_narrow.
        # Then the flattened size will be 256 * L_narrow.
        # Let's trace with input_feature_dim = 3955:
        # After MP1 (stride 2): 3955 / 2 = 1977.5 -&gt; (assume rounding or specific padding) -&gt; let's use a common approach for simplicity
        # If we use padding=1 for kernel_size=3, stride=1, output length is same as input length.
        # MP1 (kernel 2, stride 2) halves the length.
        # Sequence Length Trace (assuming integer division for pooling):
        # Input: 3955
        # Conv1 (k=3, p=1, s=1): 3955
        # MP1 (k=2, s=2): 3955 // 2 = 1977
        # Conv2 (k=3, p=1, s=1): 1977
        # Conv3 (k=3, p=1, s=1): 1977
        # Conv4 (k=3, p=1, s=1): 1977
        # MP2 (k=2, s=2): 1977 // 2 = 988

        # So, the output shape from the narrow path before flattening is (batch_size, 256, input_feature_dim / 4)
        seq_len_after_pools = max(1, input_feature_dim // 4)
        self.fc1_narrow = nn.Linear(256 * seq_len_after_pools, 100)
        self.fc2_narrow = nn.Linear(100, 50)

        # ============================================
        # 경로 2: 커널 크기 5 (Wider scale)
        # ============================================
        self.conv1_wide = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2) # padding=2 to keep length
        self.mp1_wide = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_wide = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3_wide = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4_wide = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.mp2_wide = nn.MaxPool1d(kernel_size=2, stride=2)

        # Output shape from conv4_wide is (batch_size, 256, input_feature_dim / 4)
        self.fc1_wide = nn.Linear(256 * seq_len_after_pools, 100)
        self.fc2_wide = nn.Linear(100, 50)

        # ============================================
        # 결합 후 FC 레이어
        # ============================================
        # 두 경로에서 각각 FC2 결과 (50차원)가 합쳐지므로, 총 50 + 50 = 100 차원이 됩니다.
        self.fc_combined = nn.Linear(50 + 50, 100)

        # ============================================
        # 회귀 출력 레이어
        # ============================================
        # 논문에서는 'Regression Output Layer'라고 되어 있는데, 이는 보통 마지막 FC 레이어 역할을 하거나,
        # 특정 활성화 함수(예: Sigmoid, Tanh)를 사용하여 예측 범위를 제한하는 역할을 합니다.
        # 여기서는 하나의 연속적인 값을 예측하므로, 선형 계층으로 구현합니다.
        # 만약 예측 속성 값이 0\~100 범위라면 Sigmoid + Scaling을 사용할 수 있습니다.
        # 여기서는 간단히 선형 회귀로 가정합니다.
        self.output_layer = nn.Linear(100, 1) # 1개의 속성 값을 예측한다고 가정. 만약 4개의 속성을 동시에 예측한다면 out_features=4

    def forward(self, x):
        """
        모델의 순전파(forward pass)를 정의합니다.

        Args:
            x (torch.Tensor): 입력 특징 텐서. 형태: (batch_size, input_feature_dim)

        Returns:
            torch.Tensor: 예측된 속성 값. 형태: (batch_size, 1)
        """
        # 입력 텐서 형태를 (batch_size, num_channels, sequence_length)로 변경
        # 여기서 num_channels는 1입니다.
        x = x.unsqueeze(1) # Shape: (batch_size, 1, input_feature_dim)

        # ============================================
        # 경로 1 (Narrow Scale) 처리
        # ============================================
        x_narrow = F.relu(self.conv1_narrow(x))
        x_narrow = self.mp1_narrow(x_narrow)
        x_narrow = F.relu(self.conv2_narrow(x_narrow))
        x_narrow = F.relu(self.conv3_narrow(x_narrow))
        x_narrow = F.relu(self.conv4_narrow(x_narrow))
        x_narrow = self.mp2_narrow(x_narrow)

        # Flatten for FC layers
        # x_narrow shape: (batch_size, 256, seq_len_after_mp2_narrow)
        # Flatten operation: (batch_size, 256 * seq_len_after_mp2_narrow)
        x_narrow = x_narrow.view(x_narrow.size(0), -1) # -1 추론해서 나머지 차원 계산

        # FC layers for narrow path
        x_narrow = F.relu(self.fc1_narrow(x_narrow))
        x_narrow = F.relu(self.fc2_narrow(x_narrow)) # Output shape: (batch_size, 50)

        # ============================================
        # 경로 2 (Wide Scale) 처리
        # ============================================
        x_wide = F.relu(self.conv1_wide(x))
        x_wide = self.mp1_wide(x_wide)
        x_wide = F.relu(self.conv2_wide(x_wide))
        x_wide = F.relu(self.conv3_wide(x_wide))
        x_wide = F.relu(self.conv4_wide(x_wide))
        x_wide = self.mp2_wide(x_wide)

        # Flatten for FC layers
        # x_wide shape: (batch_size, 256, seq_len_after_mp2_wide) - same seq_len as narrow path
        x_wide = x_wide.view(x_wide.size(0), -1)

        # FC layers for wide path
        x_wide = F.relu(self.fc1_wide(x_wide))
        x_wide = F.relu(self.fc2_wide(x_wide)) # Output shape: (batch_size, 50)

        # ============================================
        # 두 경로의 결과 결합
        # ============================================
        # Concatenate the outputs from both paths
        x_combined = torch.cat((x_narrow, x_wide), dim=1) # Shape: (batch_size, 50 + 50) = (batch_size, 100)

        # Combined FC layers
        x_combined = F.relu(self.fc_combined(x_combined)) # Output shape: (batch_size, 100)

        # Regression output
        predicted_rating = self.output_layer(x_combined) # Output shape: (batch_size, 1)

        return predicted_rating

# --- 모델 사용 예시 ---
if __name__ == "__main__":
    # 모델 인스턴스 생성
    # 입력 특징 차원을 논문에서 언급된 3955로 설정
    input_dim = 3955
    model = MultiScale1DCNN(input_feature_dim=input_dim)
    print("Model Architecture:")
    print(model)

    # 임의의 입력 데이터 생성 (batch_size=4)
    # 입력 형태: (batch_size, input_feature_dim)
    dummy_input = torch.randn(4, input_dim)

    # 모델을 통해 예측 수행
    with torch.no_grad(): # 그래디언트 계산 비활성화 (추론 시)
        predicted_output = model(dummy_input)

    print("\nDummy Input Shape:", dummy_input.shape)
    print("Predicted Output Shape:", predicted_output.shape) # 형태: (4, 1)
    print("Sample Prediction Output:", predicted_output)
