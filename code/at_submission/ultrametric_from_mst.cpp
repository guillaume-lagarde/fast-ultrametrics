#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>

// int dim = 2; // dimension of the space
// std::vector<std::vector<double>> points_ex {{0, 0}, {1,0},{2,1},{2,-1.99},{-2,1},{-1.5,-1}};
// std::vector<std::vector<int>> MST_ex = {{4,0},{5,0},{0,1},{1,2},{1,3}};

std::vector<std::vector<double>> points_ex {{0, 0}, {0.0112,0},{11.1,0},{12,0},{14,0},{15,0},{17,0}};
std::vector<std::vector<int>> MST_ex = {{1,0},{2,1},{2,3},{3,4},{4,5},{5,6}};


double metric(std::vector<double> x, std::vector<double> y, int dim){
  double res = 0;
  for (int i = 0; i < dim; i++){
    res += pow(x[i] - y[i],2);
  }
  return sqrt(res);
}




// disjoint set with path compression and union by size
class disjoint_set {
public:
  std::vector<std::vector<int>> forest;
  int size;
  disjoint_set(int s)
  {
    size = s;
    // initialization
    std::vector<int> v(3,1);
    forest.assign(size,v);
    for (int i = 0; i<size;i++){
      forest[i][0] = i;
      forest[i][1] = i; // successor in the set (cyclic linked chain)
      // forest[i][2] = 1; // size of the cluster, already initialized
    }
  }

  // return the next element in the partition
  int next(int i)
  {
    return forest[i][1];
  }
  
  int size_cluster(int i)
  {
    return forest[i][2];
  }
  
  int parent(int i){
    return forest[i][0];
  }

  int find(int i)
  {
    int x = i;
    while (x != parent(x)){
      forest[x][0] = forest[forest[x][0]][0]; // parent of x is now its grand parent
      x = parent(x);
    }
    return x;
  }

  int merge(int x, int y)
  {
    int rx = find(x);
    int ry = find(y);

    if (rx == ry)
      {
      }
    else
      {
	// first we change the circular list
	int nx = next(x);
	int ny = next(y);
	forest[x][1] = ny;
	forest[y][1] = nx;

	if (size_cluster(rx) < size_cluster(ry))
	  {
	    forest[ry][2] += size_cluster(rx);
	    forest[rx][0] = ry;
	  }

	else
	  {
	    forest[rx][2] += size_cluster(ry);
	    forest[ry][0] = rx;
	  }
      }
  }
    
};


// write the disjoin-set structure, just for sanity checks
void write(disjoint_set DS){
  std::cout << "\n";
  for (int i = 0; i < DS.size; i++){
    std::cout << "Cluster starting from ";
    std::cout << i;
    std::cout << " and is of size ";
    std::cout << DS.size_cluster(i);
    std::cout << "... list of elements: ";
    std::cout << i;
    std::cout << " ";
    int bla = i;
    int n = DS.next(bla);
    while (n != bla){
      std::cout << n;
      std::cout << " ";
      n = DS.next(n);
    }
    std::cout << "\n";
  }
}

// takes a set of points and a sorted MST (increasing edge weights)
class approx_cut_weights{
public:
  std::vector<double> cut_weights; // what we want to compute, a vector of double that corresponds to the approximate cut weights
  approx_cut_weights(std::vector<std::vector<double>> &points, std::vector<std::vector <int> > &MST)
  {
    int size_mst = MST.size();
    int d = points[0].size();

    // computing the cut weights
    int n_points = points.size();
    std::vector<double> distances_from_root; // max distance from a point in a cluster to the root
    distances_from_root.assign(n_points,0); // init
    cut_weights.assign(size_mst,0);
    disjoint_set DS(n_points);
    for (int i = 0; i < size_mst; i++) // loop over edges in increasing order
      {
	// int edge = index[i]; // edge we are considering
	int x = MST[i][0]; // first vertex in the edge
	int y = MST[i][1]; // second vertex in the edge
	int rx = DS.find(x); // center of cluster containing x
	int ry = DS.find(y); // center of cluster containing y
	double d_x_y = metric(points[rx],points[ry],d); // distance between the centers
	int mx = distances_from_root[rx]; // max distance from root in cluster rx
	int my = distances_from_root[ry]; // max distance from root in cluster ry
	cut_weights[i] = 5*(std::max(d_x_y,std::max(mx - d_x_y, my - d_x_y))); //real def normalement
	// cut_weights[edge] = (std::max(d_x_y,std::max(mx - d_x_y, my - d_x_y)));
	int s_rx = DS.size_cluster(rx); // size cluster rx
	int s_ry = DS.size_cluster(ry); // size cluster ry
	if (s_rx < s_ry)
	  {
	    int p = rx;
	    for (int i = 0; i < s_rx; i++){
	      distances_from_root[ry] = std::max(distances_from_root[ry],metric(points[p],points[ry],d));
	      p = DS.next(p);
	    }
	  }
	else
	  {
	    int p = ry;
	    for (int i = 0; i < s_ry; i++){
	      distances_from_root[rx] = std::max(distances_from_root[rx],metric(points[p],points[rx],d));
	      p = DS.next(p);
	    }
		  
	  }
	DS.merge(rx,ry);
      }
  
  }
};

// takes a set of points and a sorted MST (increasing edge weights)
class exact_cut_weights{
public:
  std::vector<double> cut_weights; // what we want to compute, a vector of double that corresponds to the approximate cut weights
  exact_cut_weights(std::vector<std::vector<double>> &points, std::vector<std::vector <int> > &MST)
  {
    int size_mst = MST.size();
    int d = points[0].size();

    // computing the cut weights
    int n_points = points.size();

    //std::vector<double> distances_from_root; // max distance from a point in a cluster to the root
    //distances_from_root.assign(n_points,0); // init

    cut_weights.assign(size_mst,0);
    disjoint_set DS(n_points);
    int x = 0;
    int y = 0;
    int rx = 0;
    int ry = 0;
    double d_x_y = 0;
    int s_rx = 0;
    int s_ry = 0;
    int p1 = 0;
    int p2 = 0;
    int j = 0;
    int k = 0;
    for (int i = 0; i < size_mst; i++) // loop over edges in increasing order
      {
	// int edge = index[i]; // edge we are considering
	x = MST[i][0]; // first vertex in the edge
	y = MST[i][1]; // second vertex in the edge
	
	rx = DS.find(x); // center of cluster containing x
	ry = DS.find(y); // center of cluster containing y

	d_x_y = metric(points[rx],points[ry],d); // distance between the centers

	
	cut_weights[i] = d_x_y;
	  
	// cut_weights[edge] = (std::max(d_x_y,std::max(mx - d_x_y, my - d_x_y)));
	  
	s_rx = DS.size_cluster(rx); // size cluster rx
	s_ry = DS.size_cluster(ry); // size cluster ry

	p1 = rx;
	p2 = ry;
	for (j = 0; j < s_rx; j++){
	  for (k = 0; k < s_ry; k++){
	    cut_weights[i] = std::max(cut_weights[i], metric(points[p1],points[p2],d));
	    p2 = DS.next(p2);
	  }
	  p1 = DS.next(p1);
	}
	
	// if (s_rx < s_ry)
	//   {
	//     int p = rx;
	//     for (int i = 0; i < s_rx; i++){
	//       distances_from_root[ry] = std::max(distances_from_root[ry],metric(points[p],points[ry],d));
	//       p = DS.next(p);
	//     }
	//   }
	// else
	//   {
	//     int p = ry;
	//     for (int i = 0; i < s_ry; i++){
	//       distances_from_root[rx] = std::max(distances_from_root[rx],metric(points[p],points[rx],d));
	//       p = DS.next(p);
	//     }
		  
	//   }
	DS.merge(rx,ry);
      }
  
  }
};


// Compute ultrametric from cut weights
class ultrametric {
public:
  std::vector<std::vector<int>> tree; // a tree[i][0] = left child of i, tree[i][1] = right child of i
  std::vector<double> heights; // the height of each node
  int nb_vertices;
  ultrametric(std::vector<std::vector<double>> &points, std::vector<std::vector <int> > &MST,   std::vector<double> &cut_weights)
  {
    int nb_points = points.size();
    nb_vertices = 2*nb_points-1;
    std::vector<int> v(3,-1); // -1 stands for NIL
    tree.assign(nb_vertices,v);

    for (int i = 0; i < nb_vertices;i++)
      {
	tree[i][2] = i; // root of any vertex is its own root
      }
    heights.assign(nb_vertices,0);

    // step 1. sort the edges according to their cut weights
    std::vector<int> index(cut_weights.size(), 0);
    for (int i = 0 ; i != index.size() ; i++)
      {
	index[i] = i;
      }
    std::sort(index.begin(), index.end(),
    	      [&](const int& a, const int& b) {
    		return (cut_weights[a] < cut_weights[b]);
    	      }
    	      );
    
    int node = nb_points;
    int edge = 0;
    int x = 0;
    int y = 0;
    int rx = 0;
    int ry = 0;
    for (int i = 0; i < index.size(); i++) // loop over edges in increasing order w.r.t cut weights
      {
	edge = index[i]; // edge = (x,y) we are considering
	x = MST[edge][0];
	y = MST[edge][1];
	rx = root(x);
	ry = root(y);
	tree[node][0] = rx;
	tree[node][1] = ry;
	heights[node] = cut_weights[edge]/2;
	tree[rx][2] = node;
	tree[ry][2] = node;
	node++;
      }
    
  }


  int root(int i)
  {
    int x = i;
    while (x != tree[x][2])
      {
	tree[x][2] = tree[tree[x][2]][2];
	x = tree[x][2];
      }
    return x;
  }
    
};


void print_UM(ultrametric UM)
{
  std::cout << "\n";
  std::cout << "\n";
  for (int i = 0; i< UM.nb_vertices; i++)
    {
      std::cout << "vertex ";
      std::cout << i;
      std::cout << " has height ";
      std::cout << UM.heights[i];
      std::cout << ". It has left child ";
      std::cout << UM.tree[i][0];
      std::cout << " and right child ";
      std::cout << UM.tree[i][1];
      std::cout << "\n";
    }
}


// Exact MST; to compare if we do not approximate this part of the algo.
class exact_MST{
public:
  std::vector<std::vector<int>> edges;
  exact_MST(std::vector<std::vector<double>> & points)
  {
    int size = points.size();
    int dim = points[0].size();
    std::vector<std::vector<int>> all_edges;
    std::vector<double> weights;
    for (int i = 0; i < size; i++){
      	std::cout << "\n" << i ;
      for (int j = i+1; j < size; j++){
	all_edges.push_back({i,j});
	weights.push_back(metric(points[i],points[j],dim));
      }
    }
    
    std::vector<int> index(all_edges.size(), 0);
    for (int i = 0 ; i != index.size() ; i++)
      {
	index[i] = i;
      }
    std::cout << "SORTING";
    std::sort(index.begin(), index.end(),
    	      [&](const int& a, const int& b) {
    		return (weights[a] < weights[b]);
    	      }
    	      );

    
    disjoint_set DS(size);
    int e = 0;
    int x = 0;
    int y = 0;
    for (int i = 0; i < index.size(); i++) // loop over edges in increasing order
      {
    	e = index[i]; // edge e=(x,y) we are considering
    	x = all_edges[e][0];
    	y = all_edges[e][1];
    	if (DS.find(x) != DS.find(y)){
    	  edges.push_back({x,y});
    	  DS.merge(x,y);
	}
      }
  }
};
