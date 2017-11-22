import tensorflow as tf

def abc_mul_cd(op1, op2, b, c, d):
    '''
        op1: (a, b, c)
        op2: (c, d)

    return
        result: (a, b, d)
    '''
    op1 = tf.reshape(op1, [-1, c])
    mid = tf.matmul(op1, op2)
    result =  tf.reshape(mid, [-1, b, d])
    return result

def abc_mul_c(op1, op2, b, c):
    '''
        op1: (a, b, c)
        op2: (c,)

    return
        result: (a * b)
    '''
    op1 = tf.reshape(op1, [-1, c])
    op2 = tf.reshape(op2, [-1, 1])
    mid = tf.matmul(op1, op2)
    result =  tf.reshape(mid, [-1, b])
    return result
def abc_plus_ac(op1, op2, b, c):
    '''
        op1: (a, b, c)
        op2: (a, c)
    return
        result: (a, b, c)
    '''
    op2 = tf.reshape(op2, (-1, 1, c))
    op2 = tf.tile(op2, (1, b, 1))
    return op1 + op2
