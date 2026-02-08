from conan import ConanFile


class CrewConan(ConanFile):
    generators = ("CMakeToolchain", "CMakeDeps")
    settings = ("os", "build_type", "arch", "compiler")

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        # self.requires("libtorch/2.9.1")

    def build_requirements(self):
        self.tool_requires("cmake/[<=4.2.3]")

    def layout(self):
        self.folders.generators = ""


#
