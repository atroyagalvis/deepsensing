import utils
import sys, argparse

def main(argv=None):

    if argv is None:
        argv=sys.argv[1:]

    p = argparse.ArgumentParser(description="Compares a labelled image with a reference labelled image and prints the accuracy")

    p.add_argument('predicted', action='store', default='hello', help="The predicted image to evaluate")
    p.add_argument('reference', action='store', default='world', help="The reference image to compare with")

    # Parse command line arguments
    args = p.parse_args(argv)
    
    print "comparing", args.predicted, "with", args.reference
    try:
      acc = utils.evaluate(args.predicted,args.reference)
      print "accuracy : ", acc
      conf_matrix = utils.compute_confusion_matrix(args.predicted,args.reference)
      
      for i in xrange(conf_matrix.shape[0]):
        for j in xrange(conf_matrix.shape[0]):
          print '%7s' % int(conf_matrix[i][j]),
        print ' '
    except:
      print "Error, could not compare the images"
    return 0

if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))
