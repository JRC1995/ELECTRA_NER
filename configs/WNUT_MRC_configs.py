class ELECTRA_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

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

        # loss config
        self.use_CRF = False
        self.use_sequence_label = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = True

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-MRC)'


class ELECTRA_DSC_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

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

        # loss config
        self.use_CRF = False
        self.use_sequence_label = False
        self.use_DSC = True

        # fine tune/ embd training
        self.fine_tune_style = True

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 8
        self.val_batch_size = 8
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-DSC-MRC)'


class ELECTRA_SL_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

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

        # loss config
        self.use_CRF = False
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = True

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-SL-MRC)'


class ELECTRA_CRF_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

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

        # loss config
        self.use_CRF = True
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = True

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-CRF-MRC)'


class ELECTRA_BiLSTM_natural_query_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = True
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = False
        self.use_sequence_label = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-natural-query)'


class ELECTRA_BiLSTM_SL_natural_query_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = True
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = False
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-SL-natural-query)'


class ELECTRA_BiLSTM_CRF_natural_query_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = True
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = True
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-CRF-natural-query)'


class ELECTRA_BiLSTM_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = False
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = False
        self.use_sequence_label = False
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-MRC)'


class ELECTRA_BiLSTM_SL_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = False
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = False
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-SL-MRC)'


class ELECTRA_BiLSTM_CRF_MRC_config:
    def __init__(self):
        # basic embedding config
        self.embedding_path = 'embeddings/ELECTRA/'
        self.BigTransformerDim = 1024
        self.layer_num = 24
        self.pool_type = 'mean'

        # layer aggregation / layer selection config
        self.aggregate_layers = True
        self.aggregate_num = 12
        self.select_a_particular_layer = False
        self.select_num = 18

        # query embeddings
        self.use_pretrained_query_embedding = False
        self.query_dim = 128

        # BiLSTM config
        self.use_BiLSTM = True
        self.BiLSTM_input_dropconnect = 0.0
        self.BiLSTM_hidden_dropconnect = 0.0
        self.BiLSTM_zoneout = 0.0
        self.word_dropout = 0.05
        self.hidden_size = 256
        self.BiLSTM_in_dropout = 0.5
        self.BiLSTM_out_dropout = 0.5

        # loss config
        self.use_CRF = True
        self.use_sequence_label = True
        self.use_DSC = False

        # fine tune/ embd training
        self.fine_tune_style = False

        # optimizer_settings
        self.fine_tune_lr = 2e-5
        self.lr = 1e-3
        self.wd = 1e-3
        self.use_gc = True  # gradient centralization

        # training settings
        self.total_batch_size = 32
        self.train_batch_size = 16
        self.val_batch_size = 16
        self.early_stop_patience = 3
        self.max_grad_norm = 5
        self.epochs = 100

        self.model_name = '(ELECTRA-BiLSTM-CRF-MRC)'
