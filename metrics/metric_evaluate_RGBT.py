from examples.test_metrics import evaluation_metric

# root = 'VT821'
# root = 'VT1000'
root = 'VT5000'

pred_path = "/tools/map/"+root
gt_root = '/datasets/RGBT/Test/' + root + '/GT/'
evaluation_metric(gt_root, pred_path)