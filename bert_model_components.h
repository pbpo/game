#pragma once

// This header serves as a convenience to include all individual BERT component headers.

#include "bert_embeddings.h"
#include "bert_attention.h"
#include "bert_intermediate.h"
#include "bert_output.h"
#include "bert_layer.h"
#include "bert_encoder.h"
#include "bert_pooler.h"

// Also, it's a good place for any shared utilities or enums related to these components,
// if they were to arise. For now, it's just for includes.

// The main BertConfigCpp is currently in "can_bert_for_masked_lm.h".
// If it becomes more general, it could be moved to its own file and included here too.
// For now, individual component headers include "can_bert_for_masked_lm.h" for BertConfigCpp.
