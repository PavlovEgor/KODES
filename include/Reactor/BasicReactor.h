namespace kodes {

class BasicReactor {
public:
    virtual void initialize(const ChemistryState& state) = 0;
    virtual void update(double dt) = 0;
    virtual ChemistryState getState() const = 0;
protected:
    ChemistryState state_;
};

}