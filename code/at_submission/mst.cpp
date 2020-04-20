#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <unordered_set>



// EXPERIMENT : construction MST on-the-fly (at the same time as the spanner)
class spanner{
public:
  vector<vector<int>> edges;
  vector<double> weights;
  int nb_vertices;
  spanner(vector<vector<double>> &points, vector<vector<int>> &list_LSH)
  {
    int v = 0;
    int e_c = 0; // edge counter
    nb_vertices = points.size();
    int nb_edges =nb_vertices-1;
    int counter = 0;
    disjoint_set DS(nb_vertices); 
    for (int i = 0; i < list_LSH.size(); i++){
      LSHDataStructure L = LSHDataStructure(list_LSH[i][0],list_LSH[i][1],list_LSH[i][2]);
      for (int j = 0; j < points.size(); j++){
	L.InsertPoint(j, points[j]);
      }

      	// L.Print();
      for (int i = 0; i < L.nb_bins_; i++) {
	vector<int> list_nodes;

	for (std::map<int, unordered_set<int>>::iterator it = L.bins_collection_[i].begin();
	     it != L.bins_collection_[i].end(); ++it) {
	  int counter = 0;
	  for ( auto it2 = (it->second).begin(); it2 != (it->second).end();
		++it2 )
	  //   //EXPERIMENT: random point in the cluster
	  //   {
	  //     list_nodes.push_back(*it2);
	  //   }
	  // int size_n = list_nodes.size();
	  // int random_index = rand() % size_n;
	  // int random_node = list_nodes[random_index];
	  // for (int i = 0; i < size_n; i++){
	  //   if (i != random_index){
	  //     int x = list_nodes[i];
	  //     if (DS.find(random_node) != DS.find(x)){
	  // 	edges.push_back({random_node,x});
	  // 	DS.merge(random_node,x);
	  // 	e_c++;
	  // 	if (e_c >= nb_vertices -1){
	  // 	  return;
	  // 	}
	  //     }
	  //   }
	  // }
	    // END EXPERIMENT

	    // COMMENT THIS IF YOU WANT TO USE THE EXPERIMENT ABOVE
	    {
	    if (counter == 0){
	      counter++;
	      v = *it2;
	    }
	    else{
	      if (DS.find(v) != DS.find(*it2)){
	  	edges.push_back({v,*it2});
	  	DS.merge(v,*it2);
	  	e_c++;
	  	if (e_c == nb_vertices -1){
	  	  return;
	  	}
	      }
	    }
	    }
	  // END "COMMENT THIS IF YOU WANT TO USE THE EXPERIMENT ABOVE"

	}
	      // std::cout << "\n r_ = :" << L.r_ << "\n"; 
      }
  
    }
    
  }

};

