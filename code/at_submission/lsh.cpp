#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <unordered_set>

#include "lsh.h"
// #include "ultrametric_from_mst.cpp"

using std::unordered_set;
using std::map;
using std::max;
using std::min;
using std::pair;
using std::vector;


double LSHDataStructure::SqrDist(const vector<double>& p1,
                                 const vector<double>& p2) {
  double d = 0;
  for (int i = 0; i < p1.size(); i++) {
    d += (p1[i] - p2[i]) * (p1[i] - p2[i]);
  }
  return d;
}

vector<int> LSHDataStructure::Project(const vector<double>& coordinates) {
  vector<int> projections;

  for (int i = 0; i < nb_bins_; i++) {
    int b = projectors_[i].first;
    double c = 0;
    for (int j = 0; j < projectors_[i].second.size(); j++) {
      c += projectors_[i].second[j] * coordinates[j];
    }
    projections.push_back(static_cast<int>((c + b) / r_));
  }

  return projections;
}

void LSHDataStructure::InsertPoint(int id, const vector<double> &coordinates) {
  points_.insert(pair<int, vector<double>>(id, coordinates));

  vector<int> proj = Project(coordinates);
  vector < pair < int, int >> proj_with_id;

  for (int i = 0; i < nb_bins_; i++) {
    map<int, unordered_set<int>>::iterator it_bin;
    int pos = 0;
    it_bin = bins_collection_[i].find(proj[i]);
    if (it_bin != bins_collection_[i].end()) {
      (it_bin->second).insert(id);
      pos = (it_bin->second).size()-1;
    } else {
	unordered_set<int> new_bin;
	//vector<int> new_bin;
	new_bin.insert(id);
	bins_collection_[i].insert(pair<int, unordered_set<int>>(proj[i], new_bin));
    }
    proj_with_id.push_back(pair<int, int > (proj[i], pos));
  }
  points_to_bins_.insert(pair<int,
			 vector<pair<int, int>>> (id,
						  proj_with_id));
}


void LSHDataStructure::RemovePoint(int id){
   map<int,vector<double>>::iterator it;
   it = points_.find(id);
   if(it == points_.end()) return;
   points_.erase(it);
    
   map<int,vector<pair<int,int>>>::iterator it_to_bins;
   it_to_bins = points_to_bins_.find(id);
   if(it_to_bins == points_to_bins_.end()) return;
    
   vector < pair<int,int> > proj =  it_to_bins->second;
   
   for(int i = 0; i < bins_collection_.size(); i++){
       int bin = (proj[i]).first;
       std::cout << "Erasing: " << id << " in bin " << bin << std::endl;
       // int pos = (proj[i]).second;
       map<int,unordered_set<int>>::iterator it_bins;
       it_bins = bins_collection_[i].find(bin);
       (it_bins->second).erase(id);
       // this is slow
//	for (int j = 0; j < (it_bins->second).size(); j++){
//	    if((it_bins->second)[j] == id){
//		(it_bins->second).erase((it_bins->second).begin()+j);
//		break;/
//	    }
//    }
    }
   points_to_bins_.erase(it_to_bins);
}

    
pair<int, double> LSHDataStructure::QueryPoint(int id_query,
    const vector<double>& coordinates,
    int running_time) {
  // 1. Get the projection: i.e. a list of bins b_1,...,b_nb_bins
  // 2. Consider the elements in the bins b_1,.., b_nb_bins up to a fixed budget
  // 3. Output the closest one.
  vector<int> proj = Project(coordinates);
  int nb_comparisons = 0;
  int id = -1; 
  double min_dist = std::numeric_limits<double>::infinity();

  for (int i = 0; i < nb_bins_; i++) {
    map<int, unordered_set<int>>::iterator it_bin;
    it_bin = bins_collection_[i].find(proj[i]);
    if (it_bin == bins_collection_[i].end()) continue;
    unordered_set<int> myset = (it_bin->second);
    for ( auto it = myset.begin(); it != myset.end(); ++it ){
      map<int, vector<double>>::iterator p;
      p = points_.find(*it);
      double d = SqrDist(coordinates, p->second);
      if (d < min_dist && id_query != p->first) {
	  min_dist = d;
	  id = p->first;
	  nb_comparisons++;
      }
      if (nb_comparisons > running_time) return pair<int,double>(id,min_dist);
    }
  }
  return pair<int,double>(id,min_dist);
}

// TODO(cohenaddad): Adding deletion

void LSHDataStructure::Print() {
  for (int i = 0; i < nb_bins_; i++) {
    std::cout << "Hash fun #" << i << std::endl;
    for (std::map<int, unordered_set<int>>::iterator it = bins_collection_[i].begin();
         it != bins_collection_[i].end(); ++it) {
      std::cout << "Bin #" << it->first << ":  ";
      for ( auto it2 = (it->second).begin(); it2 != (it->second).end();
	    ++it2 ){
        std::cout << *it2 << "   ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
  }
}

LSHDataStructure::LSHDataStructure(int bucket_size, int nb_bins1,
                                   int dimension) {
  nb_bins_ = nb_bins1;
  r_ = bucket_size;

  std::normal_distribution<double> distrib{0, 1};

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  
  for (int i = 0; i < nb_bins_; i++) {
      srand (time(NULL));
      int offset = rand() % r_ +1;
    vector<double> coordinates;
    coordinates.reserve(dimension);
    for (int j = 0; j < dimension; j++) {
      coordinates.push_back(distrib(generator));
    }
    projectors_.push_back(pair<int, vector<double>>(offset, coordinates));
    map<int, unordered_set<int>> new_map;
    bins_collection_.push_back(new_map);
  }
}

