class ConditionalBlendshapeNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        include_expression=True
    ):
        super(ConditionalBlendshapeNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        include_expression = 76 if include_expression else 0

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.dim_expression = include_expression# + 2 * 3 * num_encoding_fn_expr
        #self.dim_expression = 0#15# + 2 * 3 * num_encoding_fn_expr
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        #Encoding for expressions:
        #self.layers_expr = torch.nn.ModuleList()
        self.layers_expr = None
        #self.layers_expr.append( torch.nn.Linear(self.dim_expression, self.dim_expression*2))
        ##self.layers_expr.append(torch.nn.Linear(self.dim_expression*2, self.dim_expression*4))
        #self.dim_expression *= 2

        #self.layers_expr = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)
        self.layer1 = torch.nn.Linear(self.dim_xyz + self.dim_expression, hidden_size)

        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1): # was num_layers-1
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size + self.dim_expression, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid

    def forward(self, x, expr=None, **kwargs):
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
            x = xyz

        if self.dim_expression > 0:
            expr_encoding = (expr * 1/3).repeat(xyz.shape[0],1)
            #if self.layers_expr is not None:
            #   for l in self.layers_expr:
            #       expr_encoding = self.relu(l(expr_encoding))
            if self.layers_expr is not None:
               expr_encoding = self.layers_expr[0](expr_encoding)
               expr_encoding = torch.nn.functional.tanh(expr_encoding)
               #expr_encoding = self.layers_expr[1](expr_encoding)
               #expr_encoding = self.relu(expr_encoding)

            #x = torch.cat((xyz, expr.repeat(xyz.shape[0],1)), dim=1)
            x = torch.cat((xyz, expr_encoding), dim=1)
        else:
            x = xyz
        x = self.layer1(x)
        for i in range(len(self.layers_xyz )):
            if (
                i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz,expr_encoding), dim=-1)
            x = self.relu(self.layers_xyz[i](x))
        if self.use_viewdirs:
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)