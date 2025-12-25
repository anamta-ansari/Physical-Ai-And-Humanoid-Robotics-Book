---
title: Nav2 for Biped Movement
sidebar_position: 3
description: Path planning, localization, mapping, and navigating stairs and obstacles for bipedal robots
---

# Nav2 for Biped Movement

## Path planning

Path planning for bipedal robots presents unique challenges compared to wheeled or tracked robots due to their dynamic nature, balance requirements, and ability to navigate complex terrains. Nav2, the ROS 2 navigation stack, can be adapted and extended to handle the specific requirements of bipedal locomotion.

### Bipedal-Specific Path Planning Considerations

Unlike traditional mobile robots, bipedal robots must consider:

1. **Dynamic Stability**: Paths must maintain the robot's center of mass within stable regions
2. **Terrain Traversability**: Ability to step over obstacles, climb stairs, and navigate uneven surfaces
3. **Balance Constraints**: Turning and movement patterns that maintain balance
4. **Footstep Planning**: Specific placement of feet for stable locomotion
5. **Z-axis Navigation**: Ability to navigate changes in elevation (stairs, ramps)

### Nav2 Architecture for Bipedal Robots

```cpp
// Example of Nav2 plugin for bipedal path planning
#include <nav2_core/global_planner.hpp>
#include <nav2_core/footprint_collision_checker.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.h>
#include <pluginlib/class_list_macros.hpp>
#include <tf2/LinearMath/Transform.h>

namespace nav2_biped_planner
{

class BipedPathPlanner : public nav2_core::GlobalPlanner
{
public:
    BipedPathPlanner() = default;
    ~BipedPathPlanner() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
        std::string name, 
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override
    {
        // Store node parameters
        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        // Declare parameters specific to bipedal navigation
        node_->declare_parameter(name_ + ".foot_separation", 0.3);
        node_->declare_parameter(name_ + ".step_height", 0.15);
        node_->declare_parameter(name_ + ".step_length", 0.5);
        node_->declare_parameter(name_ + ".max_climb_angle", 30.0);
        node_->declare_parameter(name_ + ".balance_margin", 0.1);
        node_->declare_parameter(name_ + ".traversability_threshold", 70);

        // Get parameters
        foot_separation_ = node_->get_parameter(name_ + ".foot_separation").as_double();
        step_height_ = node_->get_parameter(name_ + ".step_height").as_double();
        step_length_ = node_->get_parameter(name_ + ".step_length").as_double();
        max_climb_angle_ = node_->get_parameter(name_ + ".max_climb_angle").as_double();
        balance_margin_ = node_->get_parameter(name_ + ".balance_margin").as_double();
        traversability_threshold_ = node_->get_parameter(name_ + ".traversability_threshold").as_int();

        RCLCPP_INFO(node_->get_logger(), "Configured Biped Path Planner for %s", name_.c_str());
    }

    void cleanup() override
    {
        RCLCPP_INFO(node_->get_logger(), "Cleaning up Biped Path Planner");
    }

    void activate() override
    {
        RCLCPP_INFO(node_->get_logger(), "Activating Biped Path Planner");
    }

    void deactivate() override
    {
        RCLCPP_INFO(node_->get_logger(), "Deactivating Biped Path Planner");
    }

    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped & start,
        const geometry_msgs::msg::PoseStamped & goal) override
    {
        nav_msgs::msg::Path path;

        // Validate start and goal positions
        if (!isStartValid(start) || !isGoalValid(goal)) {
            RCLCPP_WARN(node_->get_logger(), "Start or goal pose is invalid");
            return path;
        }

        // Convert to map coordinates
        unsigned int start_x, start_y, goal_x, goal_y;
        if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, start_x, start_y) ||
            !costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, goal_x, goal_y)) {
            RCLCPP_WARN(node_->get_logger(), "Start or goal pose is outside costmap");
            return path;
        }

        // Create traversability-aware costmap for biped
        auto traversability_costmap = createTraversabilityCostmap();

        // Plan path using modified A* that considers bipedal constraints
        auto plan = planBipedPath(start_x, start_y, goal_x, goal_y, traversability_costmap);

        // Convert plan to ROS path message
        path = convertToPath(plan);

        // Post-process path for bipedal locomotion
        path = postProcessBipedPath(path);

        return path;
    }

private:
    bool isStartValid(const geometry_msgs::msg::PoseStamped & start)
    {
        // Check if start position is suitable for bipedal robot
        double x = start.pose.position.x;
        double y = start.pose.position.y;
        double z = start.pose.position.z; // Height consideration

        unsigned int mx, my;
        if (!costmap_->worldToMap(x, y, mx, my)) {
            return false;
        }

        // Check if the start area is suitable for bipedal stance
        unsigned int index = costmap_->getIndex(mx, my);
        unsigned char cost = costmap_->getCost(index);

        // Bipedal robots can tolerate some obstacles but need stable ground
        return cost < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE &&
               isTraversableTerrain(x, y, z);
    }

    bool isGoalValid(const geometry_msgs::msg::PoseStamped & goal)
    {
        // Similar validation for goal position
        double x = goal.pose.position.x;
        double y = goal.pose.position.y;
        double z = goal.pose.position.z;

        unsigned int mx, my;
        if (!costmap_->worldToMap(x, y, mx, my)) {
            return false;
        }

        unsigned int index = costmap_->getIndex(mx, my);
        unsigned char cost = costmap_->getCost(index);

        // Goal should be in free space and traversable
        return cost < nav2_costmap_2d::FREE_SPACE &&
               isTraversableTerrain(x, y, z);
    }

    bool isTraversableTerrain(double x, double y, double z)
    {
        // Check if terrain is suitable for bipedal locomotion
        // This could include slope analysis, surface stability, etc.
        
        // Example: Check if slope is within acceptable limits
        double slope = estimateSlopeAt(x, y);
        if (slope > max_climb_angle_) {
            return false;
        }

        // Example: Check surface stability
        if (!isSurfaceStable(x, y)) {
            return false;
        }

        return true;
    }

    std::shared_ptr<nav2_costmap_2d::Costmap2D> createTraversabilityCostmap()
    {
        // Create a costmap that considers traversability for bipedal locomotion
        auto traversability_costmap = std::make_shared<nav2_costmap_2d::Costmap2D>(
            costmap_->getSizeInCellsX(),
            costmap_->getSizeInCellsY(),
            costmap_->getResolution(),
            costmap_->getOriginX(),
            costmap_->getOriginY()
        );

        // Copy base costmap and modify for bipedal considerations
        for (unsigned int i = 0; i < traversability_costmap->getSizeInCellsX(); ++i) {
            for (unsigned int j = 0; j < traversability_costmap->getSizeInCellsY(); ++j) {
                unsigned int index = traversability_costmap->getIndex(i, j);
                unsigned char base_cost = costmap_->getCost(i, j);

                // Modify cost based on traversability
                unsigned char traversability_cost = calculateTraversabilityCost(i, j, base_cost);
                traversability_costmap->setCost(i, j, traversability_cost);
            }
        }

        return traversability_costmap;
    }

    unsigned char calculateTraversabilityCost(unsigned int mx, unsigned int my, unsigned char base_cost)
    {
        // Calculate traversability cost based on bipedal constraints
        double world_x, world_y;
        costmap_->mapToWorld(mx, my, world_x, world_y);

        // Increase cost for areas that are not traversable for bipeds
        double slope = estimateSlopeAt(world_x, world_y);
        if (slope > max_climb_angle_) {
            return nav2_costmap_2d::LETHAL_OBSTACLE; // Too steep
        }

        // Increase cost for unstable surfaces
        if (!isSurfaceStable(world_x, world_y)) {
            return std::min(base_cost + 50, static_cast<unsigned char>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE));
        }

        // Increase cost for areas near drop-offs
        if (isNearDropOff(world_x, world_y)) {
            return std::min(base_cost + 75, static_cast<unsigned char>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE));
        }

        return base_cost;
    }

    std::vector<std::pair<unsigned int, unsigned int>> planBipedPath(
        unsigned int start_x, unsigned int start_y,
        unsigned int goal_x, unsigned int goal_y,
        std::shared_ptr<nav2_costmap_2d::Costmap2D> traversability_costmap)
    {
        // Implement A* path planning with bipedal-specific heuristics
        std::vector<std::pair<unsigned int, unsigned int>> path;

        // Use a modified A* algorithm that considers:
        // - Bipedal step constraints
        // - Balance maintenance
        // - Terrain traversability

        // Priority queue for A* algorithm
        std::priority_queue<PathNode, std::vector<PathNode>, ComparePathNodes> open_set;
        
        // Visited nodes tracking
        std::vector<std::vector<bool>> visited(
            traversability_costmap->getSizeInCellsX(),
            std::vector<bool>(traversability_costmap->getSizeInCellsY(), false)
        );

        // Cost tracking
        std::vector<std::vector<double>> g_score(
            traversability_costmap->getSizeInCellsX(),
            std::vector<double>(traversability_costmap->getSizeInCellsY(), std::numeric_limits<double>::infinity())
        );

        // Initialize start node
        PathNode start_node;
        start_node.x = start_x;
        start_node.y = start_y;
        start_node.g_score = 0.0;
        start_node.f_score = heuristic(start_x, start_y, goal_x, goal_y);
        
        open_set.push(start_node);
        g_score[start_x][start_y] = 0.0;

        // A* algorithm loop
        while (!open_set.empty()) {
            PathNode current = open_set.top();
            open_set.pop();

            if (current.x == goal_x && current.y == goal_y) {
                // Reconstruct path
                return reconstructPath(current, start_x, start_y);
            }

            if (visited[current.x][current.y]) {
                continue;
            }

            visited[current.x][current.y] = true;

            // Check neighbors with bipedal-specific constraints
            for (auto& neighbor : getBipedNeighbors(current, traversability_costmap)) {
                if (visited[neighbor.x][neighbor.y]) {
                    continue;
                }

                double tentative_g_score = current.g_score + distance(current, neighbor);

                if (tentative_g_score < g_score[neighbor.x][neighbor.y]) {
                    neighbor.parent_x = current.x;
                    neighbor.parent_y = current.y;
                    neighbor.g_score = tentative_g_score;
                    neighbor.f_score = tentative_g_score + heuristic(neighbor.x, neighbor.y, goal_x, goal_y);
                    
                    g_score[neighbor.x][neighbor.y] = tentative_g_score;
                    open_set.push(neighbor);
                }
            }
        }

        // If no path found
        return path;
    }

    std::vector<PathNode> getBipedNeighbors(const PathNode& current,
                                          std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap)
    {
        std::vector<PathNode> neighbors;

        // Bipedal robots have specific step patterns
        // Define possible step directions based on bipedal locomotion
        std::vector<std::pair<int, int>> step_offsets = {
            {0, 1}, {1, 0}, {0, -1}, {-1, 0},  // Cardinal directions
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, // Diagonal directions
            {0, 2}, {2, 0}, {0, -2}, {-2, 0},   // Longer steps
            {1, 2}, {2, 1}, {-1, 2}, {-2, 1},   // Diagonal longer steps
            {1, -2}, {2, -1}, {-1, -2}, {-2, -1}
        };

        for (auto& offset : step_offsets) {
            int new_x = current.x + offset.first;
            int new_y = current.y + offset.second;

            // Check bounds
            if (new_x < 0 || new_x >= static_cast<int>(costmap->getSizeInCellsX()) ||
                new_y < 0 || new_y >= static_cast<int>(costmap->getSizeInCellsY())) {
                continue;
            }

            // Check if step is valid for bipedal locomotion
            if (isValidBipedStep(current.x, current.y, new_x, new_y, costmap)) {
                PathNode neighbor;
                neighbor.x = new_x;
                neighbor.y = new_y;
                neighbors.push_back(neighbor);
            }
        }

        return neighbors;
    }

    bool isValidBipedStep(unsigned int from_x, unsigned int from_y, 
                         unsigned int to_x, unsigned int to_y,
                         std::shared_ptr<nav2_costmap_2d::Costmap2D> costmap)
    {
        // Check if the step is valid for bipedal locomotion
        // This includes checking for obstacles, drop-offs, and terrain traversability
        
        double from_world_x, from_world_y, to_world_x, to_world_y;
        costmap->mapToWorld(from_x, from_y, from_world_x, from_world_y);
        costmap->mapToWorld(to_x, to_y, to_world_x, to_world_y);

        // Check distance constraint (step length)
        double step_distance = sqrt(pow(to_world_x - from_world_x, 2) + pow(to_world_y - from_world_y, 2));
        if (step_distance > step_length_) {
            return false; // Step too long
        }

        // Check for obstacles in the step path
        if (hasObstaclesInStep(from_world_x, from_world_y, to_world_x, to_world_y)) {
            return false;
        }

        // Check for drop-offs
        if (hasDropOffInStep(from_world_x, from_world_y, to_world_x, to_world_y)) {
            return false;
        }

        // Check terrain traversability
        unsigned int mid_x = (from_x + to_x) / 2;
        unsigned int mid_y = (from_y + to_y) / 2;
        unsigned char cost = costmap->getCost(mid_x, mid_y);
        if (cost > traversability_threshold_) {
            return false;
        }

        return true;
    }

    double distance(const PathNode& a, const PathNode& b)
    {
        // Calculate distance with bipedal-specific considerations
        double dx = abs(a.x - b.x);
        double dy = abs(a.y - b.y);
        
        // Euclidean distance with potential penalties for difficult terrain
        double euclidean = sqrt(dx * dx + dy * dy);
        
        // Could add penalties based on terrain difficulty
        return euclidean;
    }

    double heuristic(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
    {
        // Heuristic function for A* (Manhattan distance with diagonal consideration)
        double dx = abs(static_cast<int>(x1 - x2));
        double dy = abs(static_cast<int>(y1 - y2));
        
        // Manhattan distance with diagonal movement consideration
        return std::max(dx, dy) * 0.9 + std::min(dx, dy) * 0.4; // Approximate diagonal cost
    }

    std::vector<std::pair<unsigned int, unsigned int>> reconstructPath(
        const PathNode& goal_node, unsigned int start_x, unsigned int start_y)
    {
        std::vector<std::pair<unsigned int, unsigned int>> path;
        PathNode current = goal_node;

        while (!(current.x == start_x && current.y == start_y)) {
            path.push_back(std::make_pair(current.x, current.y));
            if (current.parent_x == UINT_MAX) {
                // No parent, path reconstruction failed
                path.clear();
                break;
            }
            current = getParentNode(current);
        }

        path.push_back(std::make_pair(start_x, start_y));
        std::reverse(path.begin(), path.end());

        return path;
    }

    nav_msgs::msg::Path convertToPath(const std::vector<std::pair<unsigned int, unsigned int>>& plan)
    {
        nav_msgs::msg::Path path;
        path.header.frame_id = costmap_ros_->getGlobalFrameID();
        path.header.stamp = node_->now();

        for (const auto& point : plan) {
            double world_x, world_y;
            costmap_->mapToWorld(point.first, point.second, world_x, world_y);

            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position.x = world_x;
            pose.pose.position.y = world_y;
            pose.pose.position.z = 0.0; // Z will be adjusted for bipedal locomotion
            pose.pose.orientation.w = 1.0;

            path.poses.push_back(pose);
        }

        return path;
    }

    nav_msgs::msg::Path postProcessBipedPath(const nav_msgs::msg::Path& path)
    {
        // Post-process the path for bipedal locomotion requirements
        nav_msgs::msg::Path processed_path = path;

        // Add Z-height adjustments for stairs and slopes
        processed_path = adjustPathForElevationChanges(processed_path);

        // Smooth the path to reduce sharp turns that could affect balance
        processed_path = smoothBipedPath(processed_path);

        // Add footstep positions if needed
        processed_path = addFootstepPositions(processed_path);

        return processed_path;
    }

    nav_msgs::msg::Path adjustPathForElevationChanges(const nav_msgs::msg::Path& path)
    {
        // Adjust path Z-coordinates for stairs, ramps, and other elevation changes
        nav_msgs::msg::Path adjusted_path = path;

        for (auto& pose : adjusted_path.poses) {
            // Estimate Z-height at this location
            double z_height = estimateGroundHeight(pose.pose.position.x, pose.pose.position.y);
            pose.pose.position.z = z_height;
        }

        return adjusted_path;
    }

    nav_msgs::msg::Path smoothBipedPath(const nav_msgs::msg::Path& path)
    {
        // Apply smoothing algorithm suitable for bipedal locomotion
        if (path.poses.size() < 3) {
            return path; // Nothing to smooth
        }

        nav_msgs::msg::Path smoothed_path = path;

        // Simple smoothing by averaging positions (more sophisticated algorithms could be used)
        for (size_t i = 1; i < path.poses.size() - 1; ++i) {
            // Average with neighbors to smooth turns
            smoothed_path.poses[i].pose.position.x = 
                (path.poses[i-1].pose.position.x + 
                 path.poses[i].pose.position.x * 2.0 + 
                 path.poses[i+1].pose.position.x) / 4.0;
                 
            smoothed_path.poses[i].pose.position.y = 
                (path.poses[i-1].pose.position.y + 
                 path.poses[i].pose.position.y * 2.0 + 
                 path.poses[i+1].pose.position.y) / 4.0;
        }

        return smoothed_path;
    }

    nav_msgs::msg::Path addFootstepPositions(const nav_msgs::msg::Path& path)
    {
        // For bipedal robots, we might need to add specific footstep positions
        // This is a simplified approach - full footstep planning would be more complex
        return path; // Placeholder
    }

    // Helper functions for terrain analysis
    double estimateSlopeAt(double x, double y)
    {
        // Estimate terrain slope at given coordinates
        // This would typically use elevation data or other terrain analysis
        return 0.0; // Placeholder
    }

    bool isSurfaceStable(double x, double y)
    {
        // Check if surface is stable for bipedal locomotion
        // This could check for loose terrain, slippery surfaces, etc.
        return true; // Placeholder
    }

    bool isNearDropOff(double x, double y)
    {
        // Check if location is near a drop-off that bipedal robot should avoid
        return false; // Placeholder
    }

    bool hasObstaclesInStep(double from_x, double from_y, double to_x, double to_y)
    {
        // Check if there are obstacles in the path of a step
        // This would trace a line between points and check for obstacles
        return false; // Placeholder
    }

    bool hasDropOffInStep(double from_x, double from_y, double to_x, double to_y)
    {
        // Check if there are drop-offs in the path of a step
        return false; // Placeholder
    }

    double estimateGroundHeight(double x, double y)
    {
        // Estimate ground height at given coordinates
        // This would use elevation maps or other height data
        return 0.0; // Placeholder
    }

    // Path planning structures
    struct PathNode {
        unsigned int x, y;
        unsigned int parent_x = UINT_MAX, parent_y = UINT_MAX;
        double g_score = 0.0;
        double f_score = 0.0;
    };

    struct ComparePathNodes {
        bool operator()(const PathNode& a, const PathNode& b) const {
            return a.f_score > b.f_score; // Min-heap based on f_score
        }
    };

    PathNode getParentNode(const PathNode& node)
    {
        PathNode parent;
        parent.x = node.parent_x;
        parent.y = node.parent_y;
        return parent;
    }

    // ROS components
    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;

    // Bipedal-specific parameters
    double foot_separation_;
    double step_height_;
    double step_length_;
    double max_climb_angle_;
    double balance_margin_;
    int traversability_threshold_;
};

} // namespace nav2_biped_planner

PLUGINLIB_EXPORT_CLASS(nav2_biped_planner::BipedPathPlanner, nav2_core::GlobalPlanner)
```

### Footstep Planning Integration

```cpp
// Example footstep planner for bipedal navigation
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.h>

namespace nav2_biped_planner
{

class FootstepPlanner
{
public:
    FootstepPlanner() = default;
    ~FootstepPlanner() = default;

    void initialize(const std::shared_ptr<rclcpp::Node>& node)
    {
        node_ = node;
        
        // Declare parameters
        node_->declare_parameter("foot_spacing", 0.3);
        node_->declare_parameter("step_height", 0.1);
        node_->declare_parameter("max_step_width", 0.4);
        node_->declare_parameter("max_step_length", 0.6);
        node_->declare_parameter("support_polygon_margin", 0.1);
        
        // Get parameters
        foot_spacing_ = node_->get_parameter("foot_spacing").as_double();
        step_height_ = node_->get_parameter("step_height").as_double();
        max_step_width_ = node_->get_parameter("max_step_width").as_double();
        max_step_length_ = node_->get_parameter("max_step_length").as_double();
        support_polygon_margin_ = node_->get_parameter("support_polygon_margin").as_double();
    }

    std::vector<Footstep> planFootsteps(const nav_msgs::msg::Path& nominal_path,
                                       const geometry_msgs::msg::Pose& start_pose)
    {
        std::vector<Footstep> footsteps;
        
        if (nominal_path.poses.empty()) {
            return footsteps;
        }
        
        // Start with initial foot positions
        Footstep left_foot, right_foot;
        initializeFootPositions(start_pose, left_foot, right_foot);
        
        footsteps.push_back(left_foot);
        footsteps.push_back(right_foot);
        
        // Plan footsteps along the path
        for (size_t i = 0; i < nominal_path.poses.size(); ) {
            // Determine next foot placement based on path direction and balance
            auto next_left = planNextFootstep(left_foot, right_foot, nominal_path, i, true);
            auto next_right = planNextFootstep(right_foot, next_left, nominal_path, i, false);
            
            if (isValidFootstep(next_left, nominal_path)) {
                footsteps.push_back(next_left);
                left_foot = next_left;
            }
            
            if (isValidFootstep(next_right, nominal_path)) {
                footsteps.push_back(next_right);
                right_foot = next_right;
            }
            
            // Advance along path based on step length
            i = advancePathIndex(i, nominal_path, foot_spacing_);
        }
        
        return footsteps;
    }

private:
    void initializeFootPositions(const geometry_msgs::msg::Pose& start_pose,
                                Footstep& left_foot, Footstep& right_foot)
    {
        // Initialize foot positions based on starting pose
        left_foot.pose = start_pose;
        right_foot.pose = start_pose;
        
        // Offset feet laterally to create initial stance
        double half_spacing = foot_spacing_ / 2.0;
        
        // Apply lateral offset based on robot's orientation
        double yaw = tf2::getYaw(start_pose.orientation);
        double offset_x = -half_spacing * sin(yaw);
        double offset_y = half_spacing * cos(yaw);
        
        left_foot.pose.position.x += offset_x;
        left_foot.pose.position.y += offset_y;
        
        right_foot.pose.position.x -= offset_x;
        right_foot.pose.position.y -= offset_y;
    }

    Footstep planNextFootstep(const Footstep& current_support,
                             const Footstep& other_foot,
                             const nav_msgs::msg::Path& path,
                             size_t path_index,
                             bool is_left_foot)
    {
        Footstep next_step;
        
        // Determine direction of movement
        geometry_msgs::msg::Point target_point;
        if (path_index + 1 < path.poses.size()) {
            target_point = path.poses[path_index + 1].pose.position;
        } else {
            target_point = path.poses.back().pose.position;
        }
        
        // Calculate desired step location
        geometry_msgs::msg::Point current_pos = current_support.pose.position;
        double dx = target_point.x - current_pos.x;
        double dy = target_point.y - current_pos.y;
        double dist = sqrt(dx*dx + dy*dy);
        
        // Normalize direction vector
        if (dist > 0.001) { // Avoid division by zero
            dx /= dist;
            dy /= dist;
        }
        
        // Calculate step location with appropriate stride
        double stride_length = std::min(max_step_length_, dist);
        double step_x = current_pos.x + dx * stride_length;
        double step_y = current_pos.y + dy * stride_length;
        
        // Apply lateral offset to maintain balance
        double lateral_offset = foot_spacing_ / 2.0;
        if (is_left_foot) {
            // Offset to the left relative to movement direction
            double temp_x = step_x - lateral_offset * dy;
            double temp_y = step_y + lateral_offset * dx;
            step_x = temp_x;
            step_y = temp_y;
        } else {
            // Offset to the right relative to movement direction
            double temp_x = step_x + lateral_offset * dy;
            double temp_y = step_y - lateral_offset * dx;
            step_x = temp_x;
            step_y = temp_y;
        }
        
        // Set the footstep pose
        next_step.pose.position.x = step_x;
        next_step.pose.position.y = step_y;
        next_step.pose.position.z = estimateGroundHeight(step_x, step_y);
        
        // Set orientation to match path direction
        double target_yaw = atan2(dy, dx);
        next_step.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, target_yaw));
        
        // Set step timing based on path progression
        next_step.timing = calculateStepTiming(current_support, path_index);
        
        return next_step;
    }

    bool isValidFootstep(const Footstep& step, const nav_msgs::msg::Path& path)
    {
        // Check if footstep is valid based on terrain and obstacles
        double x = step.pose.position.x;
        double y = step.pose.position.y;
        
        // Check if position is on traversable terrain
        if (!isTraversableTerrain(x, y, step.pose.position.z)) {
            return false;
        }
        
        // Check if step is within step constraints
        if (isTooWideStep(step) || isTooLongStep(step)) {
            return false;
        }
        
        // Check if footstep collides with obstacles
        if (hasObstacleAt(step)) {
            return false;
        }
        
        return true;
    }

    bool isTooWideStep(const Footstep& step)
    {
        // Check if step width exceeds maximum allowable width
        // This would compare to the other foot's position
        return false; // Placeholder
    }

    bool isTooLongStep(const Footstep& step)
    {
        // Check if step length exceeds maximum allowable length
        return false; // Placeholder
    }

    bool hasObstacleAt(const Footstep& step)
    {
        // Check if there are obstacles at the footstep location
        return false; // Placeholder
    }

    double calculateStepTiming(const Footstep& current_support, size_t path_index)
    {
        // Calculate timing for next step based on path progression
        // This would consider walking speed, terrain difficulty, etc.
        return 0.5; // Placeholder (0.5 seconds per step)
    }

    size_t advancePathIndex(size_t current_index, const nav_msgs::msg::Path& path, double step_size)
    {
        // Advance the path index based on step size
        if (current_index >= path.poses.size()) {
            return current_index;
        }
        
        size_t next_index = current_index;
        double distance_traveled = 0.0;
        
        while (next_index + 1 < path.poses.size() && distance_traveled < step_size) {
            const auto& p1 = path.poses[next_index].pose.position;
            const auto& p2 = path.poses[next_index + 1].pose.position;
            
            double segment_length = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
            distance_traveled += segment_length;
            next_index++;
        }
        
        return next_index;
    }

    // Helper functions
    bool isTraversableTerrain(double x, double y, double z)
    {
        // Check if terrain at (x,y,z) is traversable for bipedal locomotion
        return true; // Placeholder
    }

    double estimateGroundHeight(double x, double y)
    {
        // Estimate ground height at coordinates (x,y)
        return 0.0; // Placeholder
    }

    // ROS components
    std::shared_ptr<rclcpp::Node> node_;

    // Parameters
    double foot_spacing_;
    double step_height_;
    double max_step_width_;
    double max_step_length_;
    double support_polygon_margin_;

    // Footstep structure
    struct Footstep {
        geometry_msgs::msg::Pose pose;
        double timing;  // Time offset from nominal step time
        int foot_index; // 0 for left, 1 for right
    };
};

} // namespace nav2_biped_planner
```

## Localization

Localization for bipedal robots requires special consideration due to their dynamic nature and the need to maintain balance during movement.

### Bipedal Localization Challenges

Bipedal robots face unique localization challenges:

1. **Dynamic Movement**: Constant motion affects sensor readings and odometry
2. **Balance Maintenance**: Localization must account for controlled falls and balance corrections
3. **Changing Body Configuration**: Joint movements affect sensor positions
4. **Terrain Interaction**: Foot-ground contact provides additional localization cues
5. **Multi-modal Sensing**: Integration of IMU, cameras, LiDAR, and proprioceptive sensors

### AMCL Extensions for Bipedal Robots

```cpp
// Example AMCL extension for bipedal robots
#include <rclcpp/rclcpp.hpp>
#include <nav2_amcl/amcl_node.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/temperature.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace nav2_biped_localization
{

class BipedAmclNode : public nav2_amcl::AmclNode
{
public:
    explicit BipedAmclNode(const rclcpp::NodeOptions& options)
    : nav2_amcl::AmclNode(options)
    {
        // Initialize bipedal-specific parameters
        this->declare_parameter("max_balance_deviation", 0.1);
        this->declare_parameter("foot_contact_timeout", 0.5);
        this->declare_parameter("imu_integration_weight", 0.7);
        this->declare_parameter("proprioceptive_weight", 0.3);
        
        max_balance_deviation_ = this->get_parameter("max_balance_deviation").as_double();
        foot_contact_timeout_ = this->get_parameter("foot_contact_timeout").as_double();
        imu_integration_weight_ = this->get_parameter("imu_integration_weight").as_double();
        proprioceptive_weight_ = this->get_parameter("proprioceptive_weight").as_double();
        
        // Subscribe to bipedal-specific sensors
        foot_contact_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "foot_contact", 10,
            std::bind(&BipedAmclNode::footContactCallback, this, std::placeholders::_1)
        );
        
        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10,
            std::bind(&BipedAmclNode::jointStateCallback, this, std::placeholders::_1)
        );
        
        RCLCPP_INFO(this->get_logger(), "Biped AMCL Node initialized");
    }

protected:
    void updatePose(
        const sensor_msgs::msg::LaserScan::SharedPtr scan,
        std::vector<Particle>& particles,
        const std::vector<double>& weights) override
    {
        // Update particle poses with bipedal-specific motion model
        auto start_time = std::chrono::steady_clock::now();
        
        // Get odometry-based motion estimate
        geometry_msgs::msg::Twist motion = getBipedMotionEstimate();
        
        // Apply motion model with bipedal-specific uncertainties
        for (size_t i = 0; i < particles.size(); ++i) {
            applyBipedMotionModel(particles[i], motion);
        }
        
        // Integrate IMU data for better orientation estimates
        integrateImuData(particles);
        
        // Use proprioceptive data (joint angles) for pose correction
        correctWithProprioception(particles);
        
        // Apply terrain-based constraints
        applyTerrainConstraints(particles);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        RCLCPP_DEBUG(this->get_logger(), "Biped pose update took %ld microseconds", duration.count());
    }

    geometry_msgs::msg::Twist getBipedMotionEstimate()
    {
        geometry_msgs::msg::Twist motion;
        
        // Estimate motion based on:
        // - Joint encoders (step detection)
        // - IMU data (rotation and acceleration)
        // - Foot contact sensors (step timing)
        // - Odometry from other sources
        
        // Calculate step-based motion estimate
        if (hasTakenStep()) {
            motion.linear.x = estimateStepDistance();
            motion.angular.z = estimateTurnAngle();
        }
        
        // Integrate IMU for rotation and acceleration
        if (latest_imu_.has_value()) {
            motion.angular.z = latest_imu_->angular_velocity.z * dt_; // Simplified
        }
        
        return motion;
    }

    void applyBipedMotionModel(Particle& particle, const geometry_msgs::msg::Twist& motion)
    {
        // Apply motion model specific to bipedal locomotion
        // This includes uncertainties related to balance, step timing, etc.
        
        double linear_uncertainty = estimateLinearUncertainty(motion.linear.x);
        double angular_uncertainty = estimateAngularUncertainty(motion.angular.z);
        
        // Add noise based on bipedal-specific uncertainties
        double noise_dx = sampleGaussian(0.0, linear_uncertainty);
        double noise_dy = sampleGaussian(0.0, linear_uncertainty);
        double noise_dtheta = sampleGaussian(0.0, angular_uncertainty);
        
        // Apply motion with noise
        double theta = particle.pose.theta;
        particle.pose.x += (motion.linear.x + noise_dx) * cos(theta) - (motion.linear.y + noise_dy) * sin(theta);
        particle.pose.y += (motion.linear.x + noise_dx) * sin(theta) + (motion.linear.y + noise_dy) * cos(theta);
        particle.pose.theta += (motion.angular.z + noise_dtheta);
        
        // Normalize angle
        particle.pose.theta = normalizeAngle(particle.pose.theta);
    }

    void integrateImuData(std::vector<Particle>& particles)
    {
        if (!latest_imu_.has_value()) {
            return;
        }
        
        // Use IMU data to correct particle orientations
        // This is particularly important for bipedal robots due to balance requirements
        
        for (auto& particle : particles) {
            // Correct orientation based on IMU gravity vector
            tf2::Quaternion imu_quat;
            tf2::fromMsg(latest_imu_->orientation, imu_quat);
            double imu_yaw = tf2::getYaw(imu_quat);
            
            // Blend estimated and measured orientation
            particle.pose.theta = imu_integration_weight_ * imu_yaw + 
                                 (1.0 - imu_integration_weight_) * particle.pose.theta;
        }
    }

    void correctWithProprioception(std::vector<Particle>& particles)
    {
        if (!latest_joint_state_.has_value()) {
            return;
        }
        
        // Use joint angle information to correct particle poses
        // This helps with accurate body configuration estimation
        
        for (auto& particle : particles) {
            // Apply corrections based on known joint angles
            // This would involve forward kinematics to determine
            // how joint configuration affects pose estimate
            applyProprioceptiveCorrection(particle);
        }
    }

    void applyTerrainConstraints(std::vector<Particle>& particles)
    {
        // Apply constraints based on terrain model
        // For example, particles that would place the robot in mid-air
        // when a foot contact is detected should be penalized
        
        if (left_foot_contact_ || right_foot_contact_) {
            for (auto& particle : particles) {
                if (isInfeasibleOnTerrain(particle)) {
                    particle.weight *= 0.1; // Significantly reduce weight
                }
            }
        }
    }

    bool hasTakenStep()
    {
        // Determine if a step has been taken based on:
        // - Joint encoder patterns
        // - IMU data (zero moment point shifts)
        // - Foot contact sensor transitions
        
        bool step_detected = false;
        
        // Check for foot contact transitions
        if (previous_left_contact_ != left_foot_contact_ || 
            previous_right_contact_ != right_foot_contact_) {
            step_detected = true;
        }
        
        // Update previous states
        previous_left_contact_ = left_foot_contact_;
        previous_right_contact_ = right_foot_contact_;
        
        return step_detected;
    }

    double estimateStepDistance()
    {
        // Estimate step distance based on:
        // - Joint encoder analysis
        // - Expected step length for current gait
        // - IMU-based displacement measurement
        
        // Placeholder implementation
        return 0.3; // Typical step distance
    }

    double estimateTurnAngle()
    {
        // Estimate turning angle based on:
        // - Step pattern (difference in left/right step locations)
        // - IMU rotation measurements
        // - Joint configuration
        
        // Placeholder implementation
        return 0.0; // No turn
    }

    double estimateLinearUncertainty(double velocity)
    {
        // Estimate uncertainty based on bipedal-specific factors
        double base_uncertainty = 0.1; // Base uncertainty
        double velocity_factor = std::abs(velocity) * 0.1; // Uncertainty increases with speed
        double balance_factor = getBalanceUncertainty(); // Uncertainty due to balance maintenance
        
        return base_uncertainty + velocity_factor + balance_factor;
    }

    double estimateAngularUncertainty(double angular_velocity)
    {
        // Estimate angular uncertainty based on bipedal factors
        double base_uncertainty = 0.05;
        double velocity_factor = std::abs(angular_velocity) * 0.05;
        
        return base_uncertainty + velocity_factor;
    }

    double getBalanceUncertainty()
    {
        // Estimate uncertainty based on how well the robot is balancing
        // This could use ZMP (Zero Moment Point) deviation or other balance metrics
        
        // Placeholder: return higher uncertainty if balance deviation is high
        return std::min(balance_deviation_ / max_balance_deviation_, 1.0) * 0.5;
    }

    bool isInfeasibleOnTerrain(const Particle& particle)
    {
        // Check if particle pose is infeasible given terrain and contact constraints
        // For example, if foot contact is detected but particle suggests foot is in air
        
        // This would involve terrain height lookup and foot positioning check
        return false; // Placeholder
    }

    void applyProprioceptiveCorrection(Particle& particle)
    {
        // Apply correction based on known joint angles
        // This involves forward kinematics to determine how joint configuration
        // affects the pose estimate
        
        // Placeholder implementation
    }

    void footContactCallback(const std_msgs::msg::Bool::SharedPtr msg)
    {
        // Store foot contact information
        // This is used for motion estimation and constraint application
        if (msg->data) {
            // Foot is in contact
            updateFootContactTime();
        }
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        latest_joint_state_ = *msg;
        
        // Analyze joint states for step detection and balance estimation
        analyzeJointStates();
    }

    void updateFootContactTime()
    {
        last_contact_time_ = this->now();
    }

    void analyzeJointStates()
    {
        // Analyze joint states for:
        // - Step detection
        // - Balance state estimation
        // - Body configuration update
        
        if (!latest_joint_state_.has_value()) {
            return;
        }
        
        // Example: Calculate Zero Moment Point (ZMP) for balance estimation
        calculateZMP();
    }

    void calculateZMP()
    {
        // Calculate Zero Moment Point to assess balance state
        // This is crucial for bipedal localization as balance affects motion
        
        // Placeholder implementation
        balance_deviation_ = 0.05; // Example value
    }

    // Callback for IMU data (inherits from parent but could extend)
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) override
    {
        latest_imu_ = *msg;
        AmclNode::imuCallback(msg); // Call parent implementation
    }

    // ROS components
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr foot_contact_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    
    // Bipedal-specific data
    std::optional<sensor_msgs::msg::JointState> latest_joint_state_;
    bool left_foot_contact_ = false;
    bool right_foot_contact_ = false;
    bool previous_left_contact_ = false;
    bool previous_right_contact_ = false;
    builtin_interfaces::msg::Time last_contact_time_;
    
    // Balance and uncertainty tracking
    double balance_deviation_ = 0.0;
    double max_balance_deviation_;
    double foot_contact_timeout_;
    double imu_integration_weight_;
    double proprioceptive_weight_;
    
    // Timing
    double dt_ = 0.1; // Placeholder for time delta
};

} // namespace nav2_biped_localization

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav2_biped_localization::BipedAmclNode)
```

## Mapping

Mapping for bipedal robots requires special consideration for 3D environments, elevation changes, and the robot's unique perspective.

### 3D Mapping for Bipedal Robots

```cpp
// Example 3D mapping node for bipedal robots
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.h>
#include <nav_msgs/msg/occupancy_grid.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

namespace nav2_biped_mapping
{

class BipedMappingNode : public rclcpp::Node
{
public:
    explicit BipedMappingNode(const rclcpp::NodeOptions& options)
    : Node("biped_mapping_node", options)
    {
        // Declare parameters
        this->declare_parameter("map_resolution", 0.05);
        this->declare_parameter("map_height", 2.0);
        this->declare_parameter("elevation_resolution", 0.1);
        this->declare_parameter("max_ground_height", 0.5);
        this->declare_parameter("min_obstacle_height", 0.1);
        this->declare_parameter("map_range", 10.0);
        this->declare_parameter("use_traversability", true);
        this->declare_parameter("traversability_method", "slope_based");
        
        // Get parameters
        map_resolution_ = this->get_parameter("map_resolution").as_double();
        map_height_ = this->get_parameter("map_height").as_double();
        elevation_resolution_ = this->get_parameter("elevation_resolution").as_double();
        max_ground_height_ = this->get_parameter("max_ground_height").as_double();
        min_obstacle_height_ = this->get_parameter("min_obstacle_height").as_double();
        map_range_ = this->get_parameter("map_range").as_double();
        use_traversability_ = this->get_parameter("use_traversability").as_bool();
        traversability_method_ = this->get_parameter("traversability_method").as_string();
        
        // Initialize TF listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Create subscribers
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "input_cloud", 10,
            std::bind(&BipedMappingNode::pointcloudCallback, this, std::placeholders::_1)
        );
        
        // Create publishers
        occupancy_grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map", 1);
        elevation_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("elevation_map", 1);
        traversability_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("traversability_map", 1);
        
        // Initialize map grid
        initializeMapGrid();
        
        RCLCPP_INFO(this->get_logger(), "Biped Mapping Node initialized");
    }

private:
    void initializeMapGrid()
    {
        // Initialize 3D occupancy grid for mapping
        map_width_cells_ = static_cast<int>(2 * map_range_ / map_resolution_);
        map_height_cells_ = static_cast<int>(2 * map_range_ / map_resolution_);
        elevation_layers_ = static_cast<int>(map_height_ / elevation_resolution_);
        
        // Initialize grids
        occupancy_grid_ = std::vector<std::vector<std::vector<int>>>(
            map_width_cells_,
            std::vector<std::vector<int>>(
                map_height_cells_,
                std::vector<int>(elevation_layers_, -1) // Unknown
            )
        );
        
        elevation_map_ = std::vector<std::vector<double>>(
            map_width_cells_,
            std::vector<double>(map_height_cells_, 0.0)
        );
        
        traversability_map_ = std::vector<std::vector<int>>(
            map_width_cells_,
            std::vector<int>(map_height_cells_, 50) // Neutral traversability
        );
        
        RCLCPP_INFO(this->get_logger(), 
            "Initialized map grid: %dx%d cells, %d elevation layers", 
            map_width_cells_, map_height_cells_, elevation_layers_);
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Transform point cloud to map frame
        sensor_msgs::msg::PointCloud2 transformed_cloud;
        if (!transformToMapFrame(*msg, transformed_cloud)) {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud to map frame");
            return;
        }
        
        // Convert to PCL format
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(transformed_cloud, *pcl_cloud);
        
        // Process the point cloud for mapping
        processPointCloudForMapping(pcl_cloud);
        
        // Update map
        updateMapFromPointCloud(pcl_cloud);
        
        // Publish updated maps
        publishMaps();
    }

    bool transformToMapFrame(const sensor_msgs::msg::PointCloud2& input, 
                           sensor_msgs::msg::PointCloud2& output)
    {
        try {
            // Get transform from sensor frame to map frame
            geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                "map", input.header.frame_id, 
                tf2::TimePoint(std::chrono::seconds(input.header.stamp.sec) +
                              std::chrono::nanoseconds(input.header.stamp.nanosec)));
            
            // Transform the point cloud
            pcl_ros::transformPointCloud("map", transform, input, output);
            output.header.frame_id = "map";
            
            return true;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform point cloud: %s", ex.what());
            return false;
        }
    }

    void processPointCloudForMapping(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        // Process point cloud to separate ground, obstacles, and traversable areas
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Separate points into different categories
        for (const auto& point : cloud->points) {
            if (isGroundPoint(point)) {
                ground_cloud->points.push_back(point);
            } else if (isObstaclePoint(point)) {
                obstacle_cloud->points.push_back(point);
            }
            // Points that don't fit either category are considered traversable free space
        }
        
        // Perform ground plane fitting
        fitGroundPlane(ground_cloud);
        
        // Analyze obstacle density and height
        analyzeObstacles(obstacle_cloud);
    }

    bool isGroundPoint(const pcl::PointXYZ& point)
    {
        // Determine if point is part of ground/surface
        // Ground points are typically within a height range from the estimated ground level
        return point.z >= -max_ground_height_ && point.z <= 0.05; // Allow slight above-ground
    }

    bool isObstaclePoint(const pcl::PointXYZ& point)
    {
        // Determine if point is part of an obstacle
        // Obstacles are typically above a minimum height
        return point.z > min_obstacle_height_;
    }

    void fitGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& ground_cloud)
    {
        // Fit a plane to ground points to estimate terrain
        if (ground_cloud->size() < 10) {
            return; // Not enough points to fit a plane
        }
        
        // Use PCL's SACSegmentation to fit a plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.05); // 5cm tolerance
        
        seg.setInputCloud(ground_cloud);
        seg.segment(*inliers, *coefficients);
        
        if (inliers->indices.size() < 10) {
            RCLCPP_WARN(this->get_logger(), "Could not estimate ground plane");
            return;
        }
        
        // Store ground plane coefficients for elevation mapping
        ground_coefficients_ = *coefficients;
        
        // Update elevation map based on fitted ground plane
        updateElevationMap(ground_cloud, coefficients);
    }

    void updateElevationMap(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& ground_cloud,
                           const pcl::ModelCoefficients::Ptr& coefficients)
    {
        // Update elevation map based on ground plane fit
        for (const auto& point : ground_cloud->points) {
            // Convert world coordinates to map indices
            int map_x = static_cast<int>((point.x + map_range_) / map_resolution_);
            int map_y = static_cast<int>((point.y + map_range_) / map_resolution_);
            
            // Check bounds
            if (map_x >= 0 && map_x < map_width_cells_ && 
                map_y >= 0 && map_y < map_height_cells_) {
                
                // Calculate elevation from ground plane
                double elevation = point.z; // Simplified - could use plane equation
                
                // Update elevation map
                elevation_map_[map_x][map_y] = elevation;
            }
        }
    }

    void analyzeObstacles(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& obstacle_cloud)
    {
        // Analyze obstacle distribution and characteristics
        for (const auto& point : obstacle_cloud->points) {
            // Convert to map coordinates
            int map_x = static_cast<int>((point.x + map_range_) / map_resolution_);
            int map_y = static_cast<int>((point.y + map_range_) / map_resolution_);
            int map_z = static_cast<int>(point.z / elevation_resolution_);
            
            // Check bounds
            if (map_x >= 0 && map_x < map_width_cells_ && 
                map_y >= 0 && map_y < map_height_cells_ &&
                map_z >= 0 && map_z < elevation_layers_) {
                
                // Mark as occupied
                occupancy_grid_[map_x][map_y][map_z] = 100; // Occupied
            }
        }
    }

    void updateMapFromPointCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        // Update the map based on the new point cloud
        // This involves ray casting to update free space and occupied space
        
        // Get robot position for ray casting
        geometry_msgs::msg::Point robot_pos = getRobotPosition();
        
        for (const auto& point : cloud->points) {
            // Perform ray casting from robot to point
            castRay(robot_pos, point);
        }
    }

    void castRay(const geometry_msgs::msg::Point& start, const pcl::PointXYZ& end)
    {
        // Cast a ray from start to end and update map along the ray
        double dx = end.x - start.x;
        double dy = end.y - start.y;
        double dz = end.z - start.z;
        
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        if (distance < 0.001) return; // Avoid division by zero
        
        double step_size = map_resolution_ / 2.0; // Finer resolution for ray casting
        int steps = static_cast<int>(distance / step_size);
        
        for (int i = 0; i < steps; ++i) {
            double t = static_cast<double>(i) / steps;
            
            double x = start.x + dx * t;
            double y = start.y + dy * t;
            double z = start.z + dz * t;
            
            // Convert to map coordinates
            int map_x = static_cast<int>((x + map_range_) / map_resolution_);
            int map_y = static_cast<int>((y + map_range_) / map_resolution_);
            int map_z = static_cast<int>(z / elevation_resolution_);
            
            // Check bounds
            if (map_x >= 0 && map_x < map_width_cells_ && 
                map_y >= 0 && map_y < map_height_cells_ &&
                map_z >= 0 && map_z < elevation_layers_) {
                
                // Update occupancy based on ray casting
                if (i == steps - 1) {
                    // Last point is the obstacle
                    occupancy_grid_[map_x][map_y][map_z] = 100; // Occupied
                } else {
                    // Intermediate points are free space
                    if (occupancy_grid_[map_x][map_y][map_z] == -1) {
                        occupancy_grid_[map_x][map_y][map_z] = 0; // Free
                    }
                }
            }
        }
    }

    void calculateTraversabilityMap()
    {
        if (!use_traversability_) return;
        
        // Calculate traversability based on elevation changes
        for (int x = 1; x < map_width_cells_ - 1; ++x) {
            for (int y = 1; y < map_height_cells_ - 1; ++y) {
                // Calculate local slope
                double slope = calculateLocalSlope(x, y);
                
                // Convert slope to traversability cost
                int traversability_cost = slopeToTraversability(slope);
                
                traversability_map_[x][y] = traversability_cost;
            }
        }
    }

    double calculateLocalSlope(int x, int y)
    {
        // Calculate local terrain slope at map cell (x, y)
        // This uses the elevation map to compute gradient
        
        if (x <= 0 || x >= map_width_cells_ - 1 || 
            y <= 0 || y >= map_height_cells_ - 1) {
            return 0.0; // Boundary case
        }
        
        // Simple 4-point gradient calculation
        double dz_dx = (elevation_map_[x+1][y] - elevation_map_[x-1][y]) / (2 * map_resolution_);
        double dz_dy = (elevation_map_[x][y+1] - elevation_map_[x][y-1]) / (2 * map_resolution_);
        
        // Calculate slope as gradient magnitude
        double slope = sqrt(dz_dx*dz_dx + dz_dy*dz_dy);
        
        // Convert to degrees
        slope = atan(slope) * 180.0 / M_PI;
        
        return slope;
    }

    int slopeToTraversability(double slope_degrees)
    {
        // Convert slope in degrees to traversability cost (0-100)
        // Lower values mean more traversable
        
        if (slope_degrees < 5.0) {
            return 10; // Very traversable
        } else if (slope_degrees < 15.0) {
            return 30; // Moderately traversable
        } else if (slope_degrees < 30.0) {
            return 60; // Difficult to traverse
        } else {
            return 100; // Not traversable
        }
    }

    geometry_msgs::msg::Point getRobotPosition()
    {
        // Get current robot position from TF or odometry
        geometry_msgs::msg::Point pos;
        
        try {
            auto transform = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero);
            
            pos.x = transform.transform.translation.x;
            pos.y = transform.transform.translation.y;
            pos.z = transform.transform.translation.z;
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Could not get robot position: %s", ex.what());
            pos.x = 0.0;
            pos.y = 0.0;
            pos.z = 0.0;
        }
        
        return pos;
    }

    void publishMaps()
    {
        // Publish occupancy grid
        auto occupancy_msg = createOccupancyGridMsg(occupancy_grid_, "map");
        occupancy_grid_pub_->publish(occupancy_msg);
        
        // Publish elevation map
        auto elevation_msg = createElevationGridMsg(elevation_map_, "map");
        elevation_map_pub_->publish(elevation_msg);
        
        // Calculate and publish traversability map
        calculateTraversabilityMap();
        auto traversability_msg = createOccupancyGridMsg(traversability_map_, "map");
        traversability_map_pub_->publish(traversability_msg);
    }

    nav_msgs::msg::OccupancyGrid createOccupancyGridMsg(
        const std::vector<std::vector<int>>& grid, const std::string& frame_id)
    {
        nav_msgs::msg::OccupancyGrid msg;
        msg.header.frame_id = frame_id;
        msg.header.stamp = this->now();
        
        msg.info.resolution = map_resolution_;
        msg.info.width = map_width_cells_;
        msg.info.height = map_height_cells_;
        msg.info.origin.position.x = -map_range_;
        msg.info.origin.position.y = -map_range_;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;
        
        msg.data.resize(map_width_cells_ * map_height_cells_);
        
        for (int y = 0; y < map_height_cells_; ++y) {
            for (int x = 0; x < map_width_cells_; ++x) {
                int index = y * map_width_cells_ + x;
                msg.data[index] = grid[x][y];
            }
        }
        
        return msg;
    }

    nav_msgs::msg::OccupancyGrid createElevationGridMsg(
        const std::vector<std::vector<double>>& elevation_grid, const std::string& frame_id)
    {
        nav_msgs::msg::OccupancyGrid msg;
        msg.header.frame_id = frame_id;
        msg.header.stamp = this->now();
        
        msg.info.resolution = map_resolution_;
        msg.info.width = map_width_cells_;
        msg.info.height = map_height_cells_;
        msg.info.origin.position.x = -map_range_;
        msg.info.origin.position.y = -map_range_;
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.w = 1.0;
        
        msg.data.resize(map_width_cells_ * map_height_cells_);
        
        // Convert elevation values to occupancy grid format (scaled to 0-100)
        for (int y = 0; y < map_height_cells_; ++y) {
            for (int x = 0; x < map_width_cells_; ++x) {
                int index = y * map_width_cells_ + x;
                // Scale elevation to 0-100 range
                int scaled_value = static_cast<int>((elevation_grid[x][y] + 1.0) * 50.0);
                scaled_value = std::max(0, std::min(100, scaled_value));
                msg.data[index] = scaled_value;
            }
        }
        
        return msg;
    }

    // ROS components
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr elevation_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr traversability_map_pub_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Map parameters
    double map_resolution_;
    double map_height_;
    double elevation_resolution_;
    double max_ground_height_;
    double min_obstacle_height_;
    double map_range_;
    bool use_traversability_;
    std::string traversability_method_;
    
    // Map grid
    int map_width_cells_;
    int map_height_cells_;
    int elevation_layers_;
    std::vector<std::vector<std::vector<int>>> occupancy_grid_; // [x][y][z]
    std::vector<std::vector<double>> elevation_map_; // [x][y]
    std::vector<std::vector<int>> traversability_map_; // [x][y]
    
    // Ground plane estimation
    pcl::ModelCoefficients ground_coefficients_;
};

} // namespace nav2_biped_mapping

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nav2_biped_mapping::BipedMappingNode)
```

## Navigating stairs and obstacles

Navigating stairs and complex obstacles requires specialized path planning and control strategies for bipedal robots.

### Stair Navigation

```cpp
// Example stair navigation module for bipedal robots
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.h>
#include <sensor_msgs/msg/laser_scan.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace nav2_biped_navigation
{

class StairNavigation
{
public:
    StairNavigation() = default;
    ~StairNavigation() = default;

    void initialize(const std::shared_ptr<rclcpp::Node>& node)
    {
        node_ = node;
        
        // Declare stair navigation parameters
        node_->declare_parameter("min_riser_height", 0.1);
        node_->declare_parameter("max_riser_height", 0.2);
        node_->declare_parameter("min_tread_depth", 0.25);
        node_->declare_parameter("max_stair_angle", 45.0);
        node_->declare_parameter("step_up_strategy", "lead_leg_first");
        node_->declare_parameter("step_down_strategy", "trail_leg_first");
        node_->declare_parameter("stair_approach_distance", 1.0);
        
        // Get parameters
        min_riser_height_ = node_->get_parameter("min_riser_height").as_double();
        max_riser_height_ = node_->get_parameter("max_riser_height").as_double();
        min_tread_depth_ = node_->get_parameter("min_tread_depth").as_double();
        max_stair_angle_ = node_->get_parameter("max_stair_angle").as_double();
        step_up_strategy_ = node_->get_parameter("step_up_strategy").as_string();
        step_down_strategy_ = node_->get_parameter("step_down_strategy").as_string();
        stair_approach_distance_ = node_->get_parameter("stair_approach_distance").as_double();
    }

    bool detectStairs(const sensor_msgs::msg::PointCloud2& pointcloud, StairConfiguration& stairs)
    {
        // Detect stairs in the point cloud data
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(pointcloud, *cloud);
        
        // Filter points in front of the robot
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = filterForStairs(cloud);
        
        if (filtered_cloud->size() < 100) {
            return false; // Not enough points to detect stairs
        }
        
        // Segment the ground plane to establish reference
        pcl::ModelCoefficients::Ptr ground_coefficients = fitGroundPlane(filtered_cloud);
        
        // Find planar segments that could represent stair treads
        std::vector<pcl::PointIndices> stair_indices = findStairCandidates(filtered_cloud);
        
        if (stair_indices.size() < 2) {
            return false; // Need at least 2 candidates to form stairs
        }
        
        // Validate and organize stair candidates
        return validateAndOrganizeStairs(stair_indices, filtered_cloud, stairs);
    }

    nav_msgs::msg::Path planStairTraversal(const StairConfiguration& stairs, 
                                         const geometry_msgs::msg::Pose& robot_pose)
    {
        nav_msgs::msg::Path path;
        
        // Plan approach to stairs
        auto approach_path = planApproachToStairs(stairs, robot_pose);
        path.poses.insert(path.poses.end(), approach_path.poses.begin(), approach_path.poses.end());
        
        // Plan stair climbing sequence
        auto climb_path = planStairClimbing(stairs);
        path.poses.insert(path.poses.end(), climb_path.poses.begin(), climb_path.poses.end());
        
        // Plan departure from stairs
        auto departure_path = planDepartureFromStairs(stairs);
        path.poses.insert(path.poses.end(), departure_path.poses.begin(), departure_path.poses.end());
        
        return path;
    }

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr filterForStairs(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        // Filter point cloud to focus on area where stairs might be
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Use pass-through filter to limit search area
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(-1.0, 3.0); // 1m behind, 3m ahead
        pass.filter(*filtered);
        
        pass.setInputCloud(filtered);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(-1.0, 1.0); // 1m left/right
        pass.filter(*filtered);
        
        pass.setInputCloud(filtered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-0.5, 1.5); // Ground to head height
        pass.filter(*filtered);
        
        return filtered;
    }

    pcl::ModelCoefficients::Ptr fitGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.05);
        
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        
        return coefficients;
    }

    std::vector<pcl::PointIndices> findStairCandidates(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
    {
        std::vector<pcl::PointIndices> stair_indices;
        
        // Use region growing or plane segmentation to find potential stair treads
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.03); // Tight threshold for flat surfaces
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = cloud->makeShared();
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        
        int nr_points = cloud_filtered->points.size();
        int num_planes = 0;
        
        while (cloud_filtered->points.size() > 0.1 * nr_points) {
            seg.setInputCloud(cloud_filtered);
            seg.segment(*inliers, *coefficients);
            
            if (inliers->indices.size() < 100) {
                // Not enough points for a valid plane
                break;
            }
            
            // Check if this plane could be a stair tread
            if (isPotentialStairTread(cloud_filtered, *inliers, *coefficients)) {
                stair_indices.push_back(*inliers);
                num_planes++;
                
                // Remove these points from the cloud
                pcl::ExtractIndices<pcl::PointXYZ> extract;
                extract.setInputCloud(cloud_filtered);
                extract.setIndices(inliers);
                extract.setNegative(true);
                extract.filter(*cloud_filtered);
            } else {
                // Just remove these points and continue
                pcl::ExtractIndices<pcl::PointXYZ> extract;
                extract.setInputCloud(cloud_filtered);
                extract.setIndices(inliers);
                extract.setNegative(true);
                extract.filter(*cloud_filtered);
            }
            
            if (num_planes > 10) { // Limit to reasonable number of planes
                break;
            }
        }
        
        return stair_indices;
    }

    bool isPotentialStairTread(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                              const pcl::PointIndices& indices,
                              const pcl::ModelCoefficients& coefficients)
    {
        // Check if the planar region could be a stair tread
        // This includes size, orientation, and height constraints
        
        // Calculate bounding box to get dimensions
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, indices.indices, min_pt, max_pt);
        
        double width = max_pt.y - min_pt.y;
        double depth = max_pt.x - min_pt.x;
        double height = max_pt.z - min_pt.z;
        
        // Check dimensions
        if (depth < min_tread_depth_ || width < 0.2) { // Minimum width for foot placement
            return false;
        }
        
        // Check orientation (should be relatively horizontal)
        double nx = std::abs(coefficients.values[0]);
        double ny = std::abs(coefficients.values[1]);
        double nz = std::abs(coefficients.values[2]);
        
        // Normal should be mostly in Z direction (horizontal surface)
        if (nz < 0.9) { // At least 90% of normal in Z direction
            return false;
        }
        
        // Check height range (should be within reasonable stair height)
        double avg_z = (min_pt.z + max_pt.z) / 2.0;
        if (avg_z < -0.5 || avg_z > 1.0) { // Stairs shouldn't be too low or too high
            return false;
        }
        
        return true;
    }

    bool validateAndOrganizeStairs(const std::vector<pcl::PointIndices>& stair_indices,
                                  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                                  StairConfiguration& stairs)
    {
        // Validate that the detected planes form a proper staircase
        // Check for regular riser heights and tread depths
        
        std::vector<StairStep> steps;
        
        for (const auto& indices : stair_indices) {
            // Calculate step properties
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cloud, indices.indices, min_pt, max_pt);
            
            StairStep step;
            step.center.x = (min_pt.x + max_pt.x) / 2.0;
            step.center.y = (min_pt.y + max_pt.y) / 2.0;
            step.center.z = (min_pt.z + max_pt.z) / 2.0;
            
            step.dimensions.x = max_pt.x - min_pt.x; // depth
            step.dimensions.y = max_pt.y - min_pt.y; // width
            step.dimensions.z = max_pt.z - min_pt.z; // height variation
            
            steps.push_back(step);
        }
        
        // Sort steps by height (Z)
        std::sort(steps.begin(), steps.end(), 
                 [](const StairStep& a, const StairStep& b) {
                     return a.center.z < b.center.z;
                 });
        
        // Validate stair regularity
        if (!validateStairRegularities(steps)) {
            return false;
        }
        
        // Calculate stair properties
        stairs.steps = steps;
        stairs.riser_height = calculateAverageRiserHeight(steps);
        stairs.tread_depth = calculateAverageTreadDepth(steps);
        stairs.number_of_steps = steps.size();
        stairs.direction = calculateStairDirection(steps);
        
        return true;
    }

    bool validateStairRegularities(const std::vector<StairStep>& steps)
    {
        if (steps.size() < 2) {
            return false;
        }
        
        // Check that steps are regularly spaced vertically
        for (size_t i = 1; i < steps.size(); ++i) {
            double riser_height = steps[i].center.z - steps[i-1].center.z;
            
            if (riser_height < min_riser_height_ || riser_height > max_riser_height_) {
                return false;
            }
        }
        
        // Check that steps are aligned horizontally
        for (size_t i = 1; i < steps.size(); ++i) {
            double depth = steps[i-1].center.x - steps[i].center.x; // Going up stairs, X decreases
            
            if (depth < min_tread_depth_ * 0.5 || depth > min_tread_depth_ * 2.0) {
                return false;
            }
        }
        
        return true;
    }

    double calculateAverageRiserHeight(const std::vector<StairStep>& steps)
    {
        if (steps.size() < 2) return 0.0;
        
        double total_height = 0.0;
        for (size_t i = 1; i < steps.size(); ++i) {
            total_height += steps[i].center.z - steps[i-1].center.z;
        }
        
        return total_height / (steps.size() - 1);
    }

    double calculateAverageTreadDepth(const std::vector<StairStep>& steps)
    {
        if (steps.size() < 2) return 0.0;
        
        double total_depth = 0.0;
        for (size_t i = 1; i < steps.size(); ++i) {
            total_depth += steps[i-1].center.x - steps[i].center.x;
        }
        
        return total_depth / (steps.size() - 1);
    }

    geometry_msgs::msg::Vector3 calculateStairDirection(const std::vector<StairStep>& steps)
    {
        geometry_msgs::msg::Vector3 direction;
        
        if (steps.size() < 2) {
            direction.x = 1.0;
            direction.y = 0.0;
            direction.z = 0.0;
            return direction;
        }
        
        // Direction is from bottom step to top step
        direction.x = steps.back().center.x - steps.front().center.x;
        direction.y = steps.back().center.y - steps.front().center.y;
        direction.z = steps.back().center.z - steps.front().center.z;
        
        // Normalize
        double length = sqrt(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);
        if (length > 0.001) {
            direction.x /= length;
            direction.y /= length;
            direction.z /= length;
        }
        
        return direction;
    }

    nav_msgs::msg::Path planApproachToStairs(const StairConfiguration& stairs,
                                           const geometry_msgs::msg::Pose& robot_pose)
    {
        nav_msgs::msg::Path approach_path;
        
        // Plan path to approach position in front of stairs
        geometry_msgs::msg::Pose approach_pose = calculateApproachPose(stairs, robot_pose);
        
        // Create a simple path to the approach position
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = "map";
        pose_stamped.header.stamp = node_->now();
        pose_stamped.pose = approach_pose;
        
        approach_path.header = pose_stamped.header;
        approach_path.poses.push_back(pose_stamped);
        
        return approach_path;
    }

    geometry_msgs::msg::Pose calculateApproachPose(const StairConfiguration& stairs,
                                                 const geometry_msgs::msg::Pose& robot_pose)
    {
        geometry_msgs::msg::Pose approach_pose;
        
        // Approach position is in front of the bottom step
        const auto& bottom_step = stairs.steps.front();
        
        approach_pose.position.x = bottom_step.center.x + stair_approach_distance_;
        approach_pose.position.y = bottom_step.center.y; // Align with stair center
        approach_pose.position.z = bottom_step.center.z; // At stair height
        
        // Orient toward the stairs
        double approach_yaw = atan2(0 - approach_pose.position.y, 
                                   bottom_step.center.x - approach_pose.position.x);
        approach_pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, approach_yaw));
        
        return approach_pose;
    }

    nav_msgs::msg::Path planStairClimbing(const StairConfiguration& stairs)
    {
        nav_msgs::msg::Path climb_path;
        climb_path.header.frame_id = "map";
        climb_path.header.stamp = node_->now();
        
        // Plan footstep sequence for climbing stairs
        for (size_t i = 0; i < stairs.steps.size(); ++i) {
            // Plan footstep for this stair
            auto step_poses = planSingleStairStep(stairs, i, true); // true = climbing up
            climb_path.poses.insert(climb_path.poses.end(), 
                                   step_poses.poses.begin(), 
                                   step_poses.poses.end());
        }
        
        return climb_path;
    }

    nav_msgs::msg::Path planSingleStairStep(const StairConfiguration& stairs,
                                          size_t step_index,
                                          bool climbing_up)
    {
        nav_msgs::msg::Path step_path;
        step_path.header.frame_id = "map";
        step_path.header.stamp = node_->now();
        
        if (step_index >= stairs.steps.size()) {
            return step_path;
        }
        
        const auto& step = stairs.steps[step_index];
        
        // Calculate footstep positions for this step
        // For climbing up: lead foot goes on the riser, trail foot follows
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = step_path.header;
        
        if (climbing_up) {
            // Lead foot placement on this step
            pose_stamped.pose.position.x = step.center.x;
            pose_stamped.pose.position.y = step.center.y;
            pose_stamped.pose.position.z = step.center.z + 0.05; // Slightly above step
            
            // Orientation to match stair direction
            double step_yaw = atan2(stairs.direction.y, stairs.direction.x);
            pose_stamped.pose.orientation = tf2::toMsg(tf2::Quaternion(0, 0, step_yaw));
        } else {
            // For climbing down
            pose_stamped.pose.position.x = step.center.x;
            pose_stamped.pose.position.y = step.center.y;
            pose_stamped.pose.position.z = step.center.z - 0.05; // Slightly below step
        }
        
        step_path.poses.push_back(pose_stamped);
        
        return step_path;
    }

    nav_msgs::msg::Path planDepartureFromStairs(const StairConfiguration& stairs)
    {
        nav_msgs::msg::Path departure_path;
        departure_path.header.frame_id = "map";
        departure_path.header.stamp = node_->now();
        
        // Plan departure from top of stairs
        if (!stairs.steps.empty()) {
            const auto& top_step = stairs.steps.back();
            
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = departure_path.header;
            
            // Position after the top step
            pose_stamped.pose.position.x = top_step.center.x - 0.5; // 0.5m beyond top step
            pose_stamped.pose.position.y = top_step.center.y;
            pose_stamped.pose.position.z = top_step.center.z;
            
            departure_path.poses.push_back(pose_stamped);
        }
        
        return departure_path;
    }

    // ROS components
    std::shared_ptr<rclcpp::Node> node_;

    // Stair navigation parameters
    double min_riser_height_;
    double max_riser_height_;
    double min_tread_depth_;
    double max_stair_angle_;
    std::string step_up_strategy_;
    std::string step_down_strategy_;
    double stair_approach_distance_;

    // Stair structures
    struct StairStep {
        geometry_msgs::msg::Point center;
        geometry_msgs::msg::Point dimensions; // x=depth, y=width, z=height variation
    };

    struct StairConfiguration {
        std::vector<StairStep> steps;
        double riser_height;
        double tread_depth;
        size_t number_of_steps;
        geometry_msgs::msg::Vector3 direction;
    };
};

} // namespace nav2_biped_navigation
```

## Conclusion

Navigation for bipedal robots presents unique challenges that require extending traditional mobile robot navigation approaches. The dynamic nature of bipedal locomotion, the need to maintain balance, and the ability to navigate complex terrains like stairs and uneven surfaces require specialized algorithms for path planning, localization, and mapping.

The integration of these capabilities into Nav2 requires modifications to consider:
- Dynamic stability constraints
- 3D mapping and elevation changes
- Specialized footstep planning
- Balance-aware localization
- Multi-modal sensor fusion

As bipedal robotics continues to advance, these navigation capabilities will become increasingly important for enabling truly autonomous bipedal robots that can navigate complex human environments effectively.