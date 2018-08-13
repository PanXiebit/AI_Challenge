import tensorflow as tf
import time
from tensorflow.contrib.linalg import LinearOperatorLowerTriangular
from model.Modules import Normalize, batchNorm_mine

"""
multi head attention
1. linearly project the queries, keys and values h times(with different, learned linear projections to d_k,d_k,d_v dimensions)
2. scaled dot product attention for each projected version of Q,K,V
3. concatenated result
4. linear projection to get final result

three kinds of usage:
1. attention for encoder
2. attention for decoder(need a mask to pay attention only known position)
3. attention as bridge of encoder and decoder
"""

def scaled_dot_product_attention(q,
                                 k,
                                 v,
                                 d_k,
                                 d_v,
                                 dropout_keep_prob):
    """ Scaled dot-product attention. One head, one spatial dimension.
    This plementation is from Google tensor2tensor.

    :param q: a tensor with shape [batch, lenght_q, d_k], we can see the sentence length of keys and query is different.
    :param k: a tensor with shape [batch, length_kv, d_k], length_kv means the sentence length of keys and values.
    :param v: a tensor with shape [batch, length_kv, d_v]
    :param bias: optional Tensor broadcastable to [batch, length_q, length_kv]
    :return: A Tensor.

    """
    with tf.variable_scope('linear-project-scaled'):
        q = tf.layers.dense(q, d_k)
        k = tf.layers.dense(k, d_k)
        v = tf.layers.dense(v, d_v)

    with tf.variable_scope('scaled_dot_product_attention'):
        scalar = tf.rsqrt(tf.to_float(q.get_shape.to_list()[2]))  # 1/sqrt(d_k)
        logits = tf.matmul(q * scalar, k, transpose_b=True)       # [batch, length_q, length_kv]

        # get probabilty
        weights = tf.nn.softmax(logits, axis=1, name='attention-weights') # [batch, length_q, length_kv]
        # drop out weights
        weights = tf.nn.dropout(weights, dropout_keep_prob)
        # output
        output = tf.matmul(weights, v)  #[batch, length_q, d_v]
        return output


def multiheadattention(q,
                       k,
                       v,
                       d_model,
                       heads,
                       keys_mask=True,
                       causality=None,
                       dropout_keep_prob=0.5,
                       is_training=True):  # is_training 这个参数在模型中必须设置
    """ multi scaled dot product attention

    :param q: A 3d tensor with shape of [batch, length_q, d_k].
    :param k: A 3d tensor with shape of [batch, lenght_kv, d_k].
    :param v:
    :param heads:An int. Number of heads.
    :param keys_mask: If true, ignore the padding
    :param causality: If true, units that reference the future are masked.
    :return:
    """
    # 1. Linear projections
    with tf.variable_scope('linear-projection-multiheads'):
        q_proj = tf.layers.dense(q, d_model) # [batch, lenght_q, d_model]
        k_proj = tf.layers.dense(k, d_model) # [batch, lenght_kv, d_model]
        v_proj = tf.layers.dense(v, d_model) # [batch, lenght_kv, d_model]

    with tf.variable_scope("multihead-attention"):
        # d_k = d_v = d_model/heads
        if d_model % heads != 0:
            raise ValueError("Key\values\query depth (%d) must be divisible by"
                             "the number of attention heads (%d)" %(d_model, heads))

        # 2. split and concat
        q_ = tf.concat(tf.split(q_proj, heads, axis=2), axis=0)  # [batch*heads, length_q, d_k]  [None, 30, 32]
        k_ = tf.concat(tf.split(k_proj, heads, axis=2), axis=0)  # [batch*heads, length_kv, d_k]
        v_ = tf.concat(tf.split(v_proj, heads, axis=2), axis=0)  # [batch*heads, length_kv, d_v]

        # 3. attention score 矩阵运算
        # outputs.shape=[batch*heads, length_q, length_kv]
        # 要理解这个矩阵运算，对一个keys的句子长度为length_kv,需要计算的其中的每一个词与query中每一个词的內积。所以最后的score是[None, length_q, lenght_kv]
    with tf.variable_scope("attention-score"):
        scalar = tf.rsqrt(d_model/heads)  # 1/sqrt(d_k)
        outputs = tf.matmul(q_*scalar, k_, transpose_b=True)   # [batch*heads, length_q, lenght_kv]

        # 4. mask 对key的屏蔽,让那些key值的unit为0的key对应的attention score极小，这样在加权计算value的时候相当于对结果不造成影响。
        # 其实也就是去掉 padding 的影响
    if keys_mask:
        with tf.variable_scope("padding-mask"):
            # `y = sign(x) = -1` if `x < 0`; 0 if `x == 0` or `tf.is_nan(x)`; 1 if `x > 0`.
            # 将第三维,也就是向量表示加起来,减小一个维度
            key_masks = tf.sign(tf.abs(tf.reduce_sum(k, axis=-1)))  # (batch, length_kv)
            # 把 key_masks 在第一个维度上重复 heads 次,目的类似于前面的 split 和 concat. 这里是针对原生的 keys,不是经过了前两步线性映射之后的 keys
            key_masks = tf.tile(key_masks, [heads, 1])    # (batch*heads, length_kv)
            # 要和前两步映射之后得到的 k_ 维度保持一致
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, q.get_shape()[1], 1])  # (batch*heads, length_q, length_kv)
            # assert (key_masks.get_shape() == outputs.get_shape())  # 好像不能这么写,有None


            # def where(condition, x=None, y=None, name=None)
            # The `condition` tensor acts as a mask that chooses, based on the value at each
            # element, whether the corresponding element / row in the output should be taken
            # from `x` (if true) or `y` (if false).
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1) # 把 keys_masks中为0的位置,对应的outputs设置的足够小
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # [batch*heads, length_q, lenght_kv]

    # Causality = Future blinding
    # causality参数告知我们是否屏蔽未来序列的信息（解码器self attention的时候不能看到自己之后的那些信息），
    # 这里即causality为True时的屏蔽操作。
    if causality:
        with tf.variable_scope("fulture-blinding"):
            # 定义一个与 outputs 后两维相同的矩阵
            diag_vals = tf.ones_like(outputs[0, :, :])  # [length_q, lenght_kv]
            tril = LinearOperatorLowerTriangular(diag_vals).to_dense()  # [length_q, lenght_kv] 得到一个三角阵，下标index大于当前行的值都变为0
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # [batch*heads, length_q, lenght_kv]

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1) # 元素全部为足够小的矩阵
            # 将 mask 中为0的位置对应的 outputs 的值设置为 MIN
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # [batch*heads, length_q, lenght_kv]

        # 将socre转换为概率
    outputs = tf.nn.softmax(outputs, axis=-1) # query 中每个词对应的 score 转化为概率

    with tf.variable_scope("query-padding-mask"):
        # Query Masking 将 padding 的影响降为最小,好奇为啥 key_mask 是选择性的?
        # query中需要mask的地方为 0(padding), 不需要mask的为 1
        query_mask = tf.sign(tf.abs(tf.reduce_sum(q, axis=-1, keepdims=False))) # [batch, lenght_q]
        query_mask = tf.tile(query_mask, [heads, 1])  # [batch*heads, length_q] # 目的是为了让query和outputs保持形状一致
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=-1), [1, 1,k.get_shape()[1]]) # [batch*heads, length_q, length_kv]
        # assert(query_mask.get_shape() == outputs.get_shape())

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(query_mask, 0), paddings, outputs) # [batch*heads, length_q, length_kv]

        # Dropout
    with tf.variable_scope("dropout-Weighted-value-res-bn"):
        outputs = tf.cond(is_training, lambda: tf.layers.dropout(outputs, dropout_keep_prob), lambda:outputs)

        # weights sum
        outputs = tf.matmul(outputs, v_)  # [batch*heads, length_q, k_v]

        # restore shape
        outputs = tf.concat(tf.split(outputs, heads, axis=0), axis=-1) #[batch,length_q, k_v*heads] = [batch, lenght_q, d_model]

        # Residual connection
        outputs += q    # [batch, lenght_q, d_model]

        # Normalize
        # tf api
        outputs = tf.layers.batch_normalization(outputs,momentum=0.99,center=True,scale=True,epsilon=0.001)
        # my implementation
        # outputs = batchNorm_mine(outputs,is_training)

    return outputs   # [batch, length_q, d_model]








