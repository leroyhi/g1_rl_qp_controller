find_package(mc_mujoco REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(g1_sim_scenario g1_sim_scenario.cpp)
target_link_libraries(g1_sim_scenario PRIVATE mc_mujoco::mc_mujoco Boost::filesystem ${PROJECT_NAME})
