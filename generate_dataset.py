import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset of optimal paths over occupancy grid maps.')
    parser.add_argument('--h', type=int, default=100,
                        help='height (# of rows) of the map')
    parser.add_argument('--w', type=int, default=100,
                        help='width (# of columns) of the map')
    parser.add_argument('--min_dist_th', type=int, default=20,
                        help='min euclidean distance (in term of cells) to keep from the start to the goal. Threshold is not guaranteed if the D* lite terminates the iterations. Anyway, the loss is evaluated considering the last point of the ground truth path as goal.')   
    parser.add_argument('--max_ds_it', type=int, default=5000,
                        help='maximum number of iteration allowed to D* lite to solve the map. Can signicantly influence completion time. My advise is to not keep it too high.')   
    parser.add_argument('--obst_margin', type=int, default=0,
                        help='# of cells to leave as margin from any obstacle. Leave it to 0 to obtain the usual D* lite behavior.')   
    parser.add_argument('--goal_margin', type=int, default=0,
                        help='# of cells to leave as margin from the goal. Leave it to 0 to obtain the usual D* lite behavior. Remeber that this the D* lite will drive you as close as possible to the goal even in case that the goal itself is on an obstacle. Please ignore and leave it to 0.')                                                                                                                           
    parser.add_argument('--n', type=int, default='10000',
                        help='# of maps to generate. The final number can be lower due to potentially unfeasible maps.')  
    parser.add_argument('--location', type=str, default='map_dataset',
                        help='height (# of rows) of the map')
    parser.add_argument('--n_process', type=int, default='1',
                        help='# of processes to be spawn')     
    
    args = parser.parse_args()

    from dataset.ds_generator import RandomMapGenerator
    rmg = RandomMapGenerator(
        args.h,
        args.w,
        args.min_dist_th,
        args.max_ds_it,
        args.obst_margin,
        args.goal_margin,
    )
    rmg.execute(args.n, args.n_process, args.location)                                           
