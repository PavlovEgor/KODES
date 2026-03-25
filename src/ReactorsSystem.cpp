
namespace kodes {

ReactorsSystem::ReactorsSystem(
    size_t n_cells,
    const vector<double>& cells_vol;
    const std::string& cantera_file,
    const std::string& solver_config_file,
    int gpu_device_id = 0
):
    ChemistryModel(cells_vol, cantera_file),
    Solver(solver_config_file)
{

}


}