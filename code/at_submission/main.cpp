#include <iostream>
#include <chrono>

#include "lsh.cpp"
#include "ultrametric_from_mst.cpp"
#include "mst.cpp"
#include "exp.cpp"
#include "csv_reader.cpp"

int epoch = 1;

// generate a random line in dimension dim
class random_line{
public:
  std::vector<std::vector<double>> points;
  std::vector<std::vector<int>> MST;
  random_line(int N, int dimension){
    std::vector<int> v(2,0);
    std::vector<double> v2(dimension,0);
    MST.assign(N-1,v);
    points.resize(N,v2);
    for (int i = 0; i < N-1; i++)
      {
	MST[i][0] = i;
	MST[i][1] = i+1;	
      }
    double start = 0;
    for (int i = 0; i < N; i++)
      {
	for (int j = 0; j < dimension-1; j++){
	  points[i][j] = 0;
	}
	points[i][dimension-1] = start;
	double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
	// start+=(r*10);
	start=(r*10000);
      }
  }
};

// 
class characteristic_points{
public:
  std::vector<std::vector<double>> points;
  characteristic_points(int N, double diameter, int dim){
    std::vector<double> v2(dim,0);
    points.resize(N,v2);
    for (int i = 0; i < N; i++)
      {
	points[i][i] = diameter/1.4142;
      }
  }
};



ultrametric clustering(std::vector<std::vector <double>> &points, double gamma, double max_dist){
  int dim = points[0].size();
  int nb_points = points.size();
  double param = pow(nb_points,1/(gamma*gamma));
  // std::cout << pow(nb_points,1/(gamma*gamma));
  // int nb_bins = std::max(2, (int) pow(nb_points,1/(gamma*gamma)) );
  int nb_bins = 5;
  // std::cout << "\n nb_bins: " << nb_bins;

  
  // TODO: need to compute list_LSH in the appropriate way
  // vector<vector<int>> list_LSH;
  // double counter = 1;
  // while (counter < max_dist){
  //   for (int i = 0; i < 1; i++){
  //     list_LSH.push_back({(int) counter,nb_bins,dim});
  //     counter = counter*1.5;
  //   }
  // }
  // END_TODO

  
  // std::cout << "\n      1. computing the MST...";

  // // FIRST CHOICE: APPROX MST
  // spanner S(points,list_LSH);

  
  // EXPERIMENT:  EXACT MST --> This improves a lot the distortion!
  exact_MST S(points);

  
  // // TO DISPLAY THE MST
  // std::cout << "\n \n \n The MST is: \n";
  // for (int i = 0; i < S.edges.size(); i++){
  //   std::cout << "\n" << S.edges[i][0] << "   " << S.edges[i][1];
  // }

  
  // std::cout << "\n      3. computing the cut weights...";

  // CHOOSE approx or exact MST
  // approx_cut_weights CW(points,S.edges);
  exact_cut_weights CW(points,S.edges);

  // // TO DISPLAY THE CUT WEIGHTS
  // std::cout << "\n \n \n The cut weights are: \n";
  // for (int i = 0; i < CW.cut_weights.size(); i++){
  //   std::cout << "\n" << i << "  " << CW.cut_weights[i];
  // }
  
  
  // std::cout << "\n      4. computing the cartesian tree...";
  ultrametric UM(points, S.edges, CW.cut_weights);
  return UM;
}



int main (){

  // // Workflow example on a random line or characteristic points
  // random_line RL(6,1);
  // // characteristic_points RL(64,1000,64);

  // std::cout << "\n computing the ultrametric..";
  // auto t1 = std::chrono::high_resolution_clock::now();
  // ultrametric UM = clustering(RL.points, 2,20000);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  // std::cout << "\n duration: " << duration;
  // // ultrametric UM = clustering(P_test, 0.1);
  // // print UM
  // std::cout << "\n \n \n The ultrametric is: \n";
  // print_UM(UM);
  // for (int i =0; i < RL.points.size(); i++){
  //   std::cout << "\n point " << i << ": ";
  //   for (int j = 0; j < RL.points[i].size(); j++){
  //     std::cout << "  " << RL.points[i][j];
  //   }
  // }

  // std::cout << "\n computing the distortion...";
  // double dist = distortion(RL.points, UM);
  // // double dist = distortion(P_test, UM);
  // std::cout << "\n The distortion is: " << dist;

  //   lca_farach LCA(UM.tree,UM.nb_vertices);

  //   for (int i = 0; i < UM.nb_vertices; i++){
  //     for (int j = i+1; j < UM.nb_vertices; j++){
  // 	std::cout << "\n LCA between "<< i << " and " << j << "   " <<LCA.query(i,j);
  //     }
  //   }
    
//   for (int i = 0; i < LCA.eulerian_tour.size(); i++){
//     std::cout << "\n " << i << "   " << LCA.eulerian_tour[i][0] << "   " << LCA.eulerian_tour[i][1];
// }
  
  // for (int i = 0; i < LCA.index.size(); i++){
  //   std::cout << "\n" << i << "    " << LCA.index[i];
  // }

  // for (int i = 0; i < LCA.sparse_table.size();i++)
  //   {
  //     for (int j = 0; j < LCA.sparse_table[i].size(); j++)
  // 	{
  // 	  std::cout << "\n  " << "i : "<< i << "  j :  " << j << "   "<< LCA.sparse_table[i][j];
  // 	}
  // }

  // Workflow exampe on a real dataset
  
  std::string fileName = "dataset/SHUTTLE.csv";
  std::cout << "\n" << fileName; 
  double rescale = 45;

  // std::cout << "\n parsing data...";
  CSVReader csv(fileName, rescale);

  std::cout << "\n Number points: " << csv.points.size() << "\n";
  std::cout << "\n N = " << epoch << "\n";


  double time_average = 0;
  double distortion_average = 0;
  for (int i = 0; i < epoch; i++){
    std::cout << "\n Trial " << i;
    // std::cout << "\n computing the ultrametric..";
    auto t1 = std::chrono::high_resolution_clock::now();
    ultrametric UM = clustering(csv.points, 2,200);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    time_average+=duration;
    // std::cout << "\n duration: " << duration;
  // ultrametric UM = clustering(P_test, 0.1);
  // print UM
  // std::cout << "\n \n \n The ultrametric is: \n";
  // print_UM(UM);
  // for (int i =0; i < csv.points.size(); i++){
  //   std::cout << "\n point " << i << ": ";
  //   for (int j = 0; j < csv.points[i].size(); j++){
  //     std::cout << "  " << csv.points[i][j];
  //   }
  // }



    // //    COMMENT THIS IF YOU JUST WANT THE AVERAGE TIME
    // std::cout << "\n computing the distortion...";
    // double dist = distortion(csv.points, UM);
    // // double dist = distortion(P_test, UM);
    // // std::cout << "\n The distortion is: " << dist;
    // distortion_average+=dist;
    // //END COMMENT THIS

  }
  // // COMMENT THIS IF YOU JUST WANT THE AVERAGE TIME
  // std::cout << "\n dist average: " << distortion_average/epoch;
  // // END COMMENT THIS
  std::cout << "\n time average: " << time_average/epoch;

  // TO PRINT SOME INFO
  
  // for (int i = 0; i < S.edges.size(); i++){
  //   std::cout << S.edges[i][0] << "   " << S.edges[i][1] << "\n";
  // }
  
  // // MST M(S);
  // for (int i = 0; i < M.edges.size();i++){
  //   std::cout << "#edge " << i << ":   " << M.edges[i][0] << "   " << M.edges[i][1] << "\n";
  // }  
}
