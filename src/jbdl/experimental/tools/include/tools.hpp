int multiply(int i, int j);

class Pet
{
    public:
        Pet(const std::string &name, int hunger): name(name), hunger(hunger) {};
        ~Pet() {};
        void go_for_a_walk();
        const std::string &get_name() const;
        int get_hunger() const;

    private:
        std::string name;
        int hunger;
};

