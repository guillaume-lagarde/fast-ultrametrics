double metric2(std::vector<double> x, std::vector<double> y, int dim){
  double res = 0;
  for (int i = 0; i < dim; i++){
    res += pow(x[i] - y[i],2);
  }
  return sqrt(res);
}

std::vector<int> lca_aux(std::vector<std::vector<int>> &tree, int pos){
  std::vector<int> res;
  if (tree[pos][0] == -1){
    res.push_back(pos);
    return res;
  }
  else{
    std::vector<int> left = lca_aux(tree, tree[pos][0]);
    std::vector<int> right = lca_aux(tree, tree[pos][1]);
    for (int i = 0; i < left.size(); i++){
      res.push_back(left[i]);
    }
    for (int j = 0; j < right.size(); j++){
      res.push_back(right[j]);
    }
    return res;
  }
}

void print_vect(std::vector<int> v){
  std::cout << "\n print vector \n";
  for (int i = 0; i < v.size(); i++){
    std::cout << v[i] << "   ";
  }
}


std::vector<std::vector<int>> lca(std::vector<std::vector<int>> &tree, int nb_points){
  std::vector<std::vector<int>> M;
  std::vector<int> v(nb_points,-1); // 
  M.assign(nb_points,v);
  for (int i = tree.size()-1; i >= nb_points; i--){
    // std::cout << "lca " << i << "\n";
    std::vector<int> left = lca_aux(tree, tree[i][0]);
    // print_vect(left);
    std::vector<int> right = lca_aux(tree, tree[i][1]);
    // print_vect(right);
    for (int a = 0; a < left.size(); a++){
      for (int b = 0; b < right.size(); b++){
	M[left[a]][right[b]] = i;
	M[right[b]][left[a]] = i;
      }
    }
  }
  return M;

}

double distortion2(std::vector<std::vector<double>> &points, ultrametric &UM)
{
  int nb_points = points.size();
  std::vector<std::vector<int>> M = lca(UM.tree, nb_points);
  double min_cheat = 10;
  double dist = 0;
  for (int i = 0; i < nb_points; i++){
    // std::cout << "\n" << i << "   " << nb_points;
    for (int j = i+1; j < nb_points; j++){
      int l = M[i][j];
      double distance_points = metric2(points[i],points[j],points[i].size());
      double ratio =2*UM.heights[l]/distance_points;
      if (distance_points > 0.01){
	min_cheat = std::min(min_cheat,ratio);
	// double temp = std::fabs(2*UM.heights[l]/distance_points);
	dist = std::max(ratio,dist);
      }
    }
  }
  std::cout << "\n \n min_dist: " << min_cheat;
  return dist/min_cheat;
}


class lca_farach {
public:
  std::vector<std::vector<int>> eulerian_tour;
  std::vector<int> index; //
  std::vector<std::vector<int>> sparse_table;
  lca_farach(std::vector<std::vector<int>> &tree, int nb_vertices)
  {
    int root = tree.size()-1;
    init(tree,root,0);

    index.assign(nb_vertices,-1);
    for (int i = 0; i < eulerian_tour.size(); i++){
      index[eulerian_tour[i][0]] = i;
    }

    int size_eulerian = eulerian_tour.size();
    
    int max_exponent = int(log2(nb_vertices)+1);
    std::cout << "\n MAX EXPO " << max_exponent;


    std::vector<int> v(max_exponent+1,-1); // 
    sparse_table.assign(size_eulerian,v);

    // We init the sparse table
    for (int i = 0; i < size_eulerian; i++){
      sparse_table[i][0] = eulerian_tour[i][0];
    }

    for (int j = 1; j < sparse_table[0].size(); j++){
      for (int i = 0; i < sparse_table.size(); i++){
    	// sparse_table[i][j] = eulerian_tour[i][0];
	int m = std::min(i+pow(2,j-1), (double) (sparse_table.size()-1));
    	if (eulerian_tour[index[sparse_table[i][j-1]]][1] < eulerian_tour[index[sparse_table[m][j-1]]][1])
    	  {
    	    sparse_table[i][j] = sparse_table[i][j-1];
    	  }
    	else
    	  {
    	    sparse_table[i][j] = sparse_table[m][j-1];
    	  }
	
      }
      
    }
    
  }

  int query(int x,int y)
  {
    int index_x = index[x];
    int index_y =index[y];
    if (index_x < index_y){
      int sz = floor(log2(index_y - index_x + 1));
      if (eulerian_tour[index[sparse_table[index_x][sz]]][1] < eulerian_tour[index[sparse_table[index_y - pow(2,sz)+1][sz]]][1]){
	return sparse_table[index_x][sz];
      }
      else
	{
	  return sparse_table[index_y - pow(2,sz)+1][sz];
	}
    }

    else{
      int sz = floor(log2(index_x - index_y + 1));
      if (eulerian_tour[index[sparse_table[index_y][sz]]][1] < eulerian_tour[index[sparse_table[index_x - pow(2,sz)+1][sz]]][1]){
	return sparse_table[index_y][sz];;
      }
      else
	{
	  return sparse_table[index_x - pow(2,sz)+1][sz];
	}
    }
    

  }


  // init(node,level)
  void init(std::vector<std::vector<int>> &tree, int vertex, int level)
  {
    if (vertex != -1){
      std::vector<int> ET(2,0);
      std::vector<int> I(2,0);
      ET[0] = vertex;
      ET[1] = level;
      eulerian_tour.push_back(ET);

      if (tree[vertex][0] != -1){
	init(tree,tree[vertex][0],level+1);
      

      eulerian_tour.push_back(ET);
    }
      if (tree[vertex][1] != -1){
      init(tree,tree[vertex][1],level+1);

      eulerian_tour.push_back(ET);
      }
	
    }
    
  }

};


// EXPERIMENT WITH FARACH
double distortion(std::vector<std::vector<double>> &points, ultrametric &UM)
{
  int nb_points = points.size();
  lca_farach LCA(UM.tree,UM.nb_vertices);
  double min_cheat = 10;
  double dist = 0;
  for (int i = 0; i < nb_points; i++){
    // std::cout << "\n" << i << "   " << nb_points;
    for (int j = i+1; j < nb_points; j++){
      int l = LCA.query(i,j);
      double distance_points = metric2(points[i],points[j],points[i].size());
      double ratio =2*UM.heights[l]/distance_points;
      if (distance_points > 0.00001){
	min_cheat = std::min(min_cheat,ratio);
	// double temp = std::fabs(2*UM.heights[l]/distance_points);
	dist = std::max(ratio,dist);
      }
    }
  }
  std::cout << "\n \n min_dist: " << min_cheat;
  return dist/min_cheat;
}



