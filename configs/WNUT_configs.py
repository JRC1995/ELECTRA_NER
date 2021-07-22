class BERT_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/BERT/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(BERT)'


class BERT_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/BERT/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(BERT-CRF)'


class BERT_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/BERT/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(BERT-BiLSTM-CRF)'


class BERT_w2v_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/BERT/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = True
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(BERT-w2v-BiLSTM-CRF)'


class BERT_extra_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/BERT/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(BERT-extra-BiLSTM-CRF)'


class ELECTRA_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA)'


class ELECTRA_fine_tune_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = True

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-CRF)'


class ELECTRA_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-CRF)'


class ELECTRA_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-CRF)'


class ELECTRA_w2v_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = True
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-w2v-BiLSTM-CRF)'


class ELECTRA_extra_BiLSTM_DSC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = False
        self.use_DSC = True

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-extra-BiLSTM-DSC)'


class ELECTRA_extra_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-extra-BiLSTM-CRF)'


class ELECTRA_w2v_extra_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = True
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-w2v-extra-BiLSTM-CRF)'


class ELECTRA_extra_CRF_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-extra-CRF)'


class ELECTRA_extra_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-extra)'


class CSE_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE)'


class CSE_CRF_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = False
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE-CRF)'


class CSE_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE-BiLSTM-CRF)'


class CSE_w2v_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = True
        self.use_fasttext = False
        self.use_pos_tags = False
        self.use_char_feats = False
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE-w2v-BiLSTM-CRF)'


class CSE_extra_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = False
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE-extra-BiLSTM-CRF)'


class CSE_w2v_extra_BiLSTM_CRF_config:
    def __init__(self):
        # basic embedding config
        self.use_backward = True
        self.use_forward = True
        self.embed_dim = (int(self.use_backward) + int(self.use_forward)) * 2048

        # extra_features
        self.phono_feats_dim = 22
        self.use_w2v = True
        self.use_fasttext = False
        self.use_pos_tags = True
        self.use_char_feats = True
        self.ipa_dim = 32
        self.pos_dim = 16
        self.char_cnn_kernels = (3, 5, 7)
        self.char_cnn_channels = 32

        # layer aggregation / layer selection config
        self.aggregate_layers = False
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0

        # multi task
        self.use_MTL = False

        # loss config
        self.use_CRF = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(CSE-w2v-extra-BiLSTM-CRF)'
